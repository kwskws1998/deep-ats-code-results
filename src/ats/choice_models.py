"""Conditional choice likelihoods for original ATS intervals and an explicit binary-task adaptation.

The observe-or-bet adaptation uses a binary hidden Markov model, not the scalar
Gaussian filter. It is a separate hypothesis that cannot validate the original
Gaussian sampling policy by itself. All choice features precede feedback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import ndtr

from ats.human_data import ObserveBlock
from ats.fit_support import select_optimization


ACTIONS = (-1, 0, 1)
CHOICE_MODELS = (
    "constant",
    "bayes_value",
    "bayes_time",
    "ats_binary_parameter_ig",
    "bayes_state_ig",
    "leaky_threshold",
)
LAPSE = 0.02


@dataclass(frozen=True)
class ChoiceFit:
    """Maximum-likelihood estimates and numerical diagnostics from training choices only."""

    model: str
    parameters: np.ndarray
    training_nll: float
    success: bool
    boundary_parameters: tuple[int, ...]
    starts: int


def categorical_probabilities(
    design: np.ndarray, parameters: np.ndarray, lapse: float = LAPSE
) -> np.ndarray:
    """Mix a stable softmax with a fixed uniform lapse, without fitting a redundant utility scale."""
    if (
        design.ndim != 3
        or design.shape[2] != len(parameters)
        or not np.isfinite(design).all()
        or not np.isfinite(parameters).all()
    ):
        raise ValueError("Invalid choice design or parameters.")
    if not 0 <= lapse < 1:
        raise ValueError("lapse must be in [0, 1).")
    logits = np.einsum("nkp,p->nk", design, parameters)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return (1 - lapse) * probability + lapse / probability.shape[1]


def choice_objective(
    parameters: np.ndarray,
    design: np.ndarray,
    choices: np.ndarray,
    lapse: float = LAPSE,
) -> tuple[float, np.ndarray]:
    """Return the exact conditional negative log likelihood and its analytic parameter gradient."""
    logits = np.einsum("nkp,p->nk", design, parameters)
    logits -= logits.max(axis=1, keepdims=True)
    soft = np.exp(logits)
    soft /= soft.sum(axis=1, keepdims=True)
    probability = (1 - lapse) * soft + lapse / soft.shape[1]
    indices = np.arange(len(choices))
    chosen = probability[indices, choices]
    weight = (1 - lapse) * soft[indices, choices] / chosen
    residual = soft.copy()
    residual[indices, choices] -= 1
    gradient = np.einsum("nk,nkp->p", weight[:, None] * residual, design)
    return float(-np.log(chosen).sum()), gradient


def fit_design(
    model: str,
    design: np.ndarray,
    choices: np.ndarray,
    bounds: tuple,
    starts: tuple | None = None,
) -> ChoiceFit:
    """Use deterministic multistart bounded likelihood optimization and expose boundary solutions."""
    choices = np.asarray(choices, dtype=int)
    if (
        len(choices) != len(design)
        or not len(choices)
        or np.any((choices < 0) | (choices >= design.shape[1]))
    ):
        raise ValueError("Training choices must index the supplied alternatives.")
    if starts is None:
        middle = np.array([(a + b) / 2 for a, b in bounds])
        starts = (
            np.clip(np.zeros(len(bounds)), *np.asarray(bounds).T),
            middle,
            np.array([a + 0.2 * (b - a) for a, b in bounds]),
        )
    results = [
        minimize(
            choice_objective,
            start,
            args=(design, choices),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 600},
        )
        for start in starts
    ]
    best = select_optimization(
        results, choice_objective, bounds, args=(design, choices), jac=True
    )
    boundary = tuple(
        i
        for i, (value, (low, high)) in enumerate(zip(best.x, bounds))
        if min(value - low, high - value) <= 1e-5 * max(1, high - low)
    )
    return ChoiceFit(
        model, best.x, float(best.fun), bool(best.success), boundary, len(starts)
    )


def binary_entropy(probability: np.ndarray | float) -> np.ndarray:
    """Compute Bernoulli entropy in nats, including deterministic limiting cases."""
    p = np.clip(probability, 1e-15, 1 - 1e-15)
    return -p * np.log(p) - (1 - p) * np.log1p(-p)


class BinaryNoiseBelief:
    """Finite-grid joint inference over state switching probability and emission reliability."""

    def __init__(
        self, hazards=(0.005, 0.025, 0.05, 0.1, 0.2), reliabilities=(0.6, 0.7, 0.8, 0.9)
    ):
        h, r = np.meshgrid(hazards, reliabilities, indexing="ij")
        self.hazards, self.reliabilities = h.ravel(), r.ravel()
        if np.any((self.hazards < 0) | (self.hazards > 0.5)) or np.any(
            (self.reliabilities <= 0.5) | (self.reliabilities >= 1)
        ):
            raise ValueError("Invalid binary transition or emission grid.")
        self.joint = np.full((h.size, 2), 1 / (2 * h.size))

    def features(self) -> np.ndarray:
        """Describe beliefs before the next action; no current outcome is available here."""
        mass = self.joint.sum(axis=1)
        positive = self.joint[:, 1] / np.maximum(mass, 1e-300)
        outcome_by_parameter = self.reliabilities * positive + (
            1 - self.reliabilities
        ) * (1 - positive)
        outcome = float(mass @ outcome_by_parameter)
        entropy = float(binary_entropy(outcome))
        parameter_ig = max(
            0.0, entropy - float(mass @ binary_entropy(outcome_by_parameter))
        )
        conditional_state_ig = max(
            0.0,
            float(
                mass
                @ (
                    binary_entropy(outcome_by_parameter)
                    - binary_entropy(self.reliabilities)
                )
            ),
        )
        return np.array(
            [
                2 * outcome - 1,
                parameter_ig / np.log(2),
                conditional_state_ig / np.log(2),
                float(mass @ self.hazards),
                float(mass @ self.reliabilities),
            ]
        )

    def advance(self, feedback: int = 0) -> None:
        """Condition on observed feedback only, then propagate one task trial even after a miss."""
        if feedback not in (-1, 0, 1):
            raise ValueError("Invalid feedback.")
        if feedback:
            emission_positive = np.column_stack(
                (1 - self.reliabilities, self.reliabilities)
            )
            self.joint *= emission_positive if feedback == 1 else 1 - emission_positive
            self.joint /= self.joint.sum()
        self.joint = (1 - self.hazards[:, None]) * self.joint + self.hazards[
            :, None
        ] * self.joint[:, ::-1]


def block_features(block: ObserveBlock) -> np.ndarray:
    """Replay observed history causally, preserving all task-clock steps and no-feedback bets."""
    belief = BinaryNoiseBelief()
    rows = []
    for choice, feedback in zip(block.choices, block.feedback):
        rows.append(belief.features())
        belief.advance(int(feedback) if choice == 0 else 0)
    return np.asarray(rows)


def choice_design(features: np.ndarray, model: str) -> tuple[np.ndarray, tuple]:
    """Match information and deadline models at four parameters, including a common outcome-color bias."""
    count = len(features)
    if model == "constant":
        design = np.zeros((count, 3, 2))
        design[:, 0, 0], design[:, 1, 1] = 1, 1
        return design, ((-8, 8), (-8, 8))
    dimensions = 3 if model == "bayes_value" else 4
    design = np.zeros((count, 3, dimensions))
    design[:, 0, 0], design[:, 2, 0] = -features[:, 0], features[:, 0]
    design[:, 1, 1] = 1
    design[:, 0, 2], design[:, 2, 2] = -1, 1
    bounds = ((0, 30), (-8, 8), (-5, 5))
    if dimensions == 4:
        remaining = (49 - np.arange(count)) / 49
        if model == "bayes_time":
            design[:, 1, 3] = remaining
            bounds += ((-12, 12),)
        elif model in ("ats_binary_parameter_ig", "bayes_state_ig"):
            column = 1 if model == "ats_binary_parameter_ig" else 2
            design[:, 1, 3] = remaining * features[:, column]
            bounds += ((0, 50),)
        else:
            raise ValueError(f"Unknown choice model: {model}")
    return design, bounds


def choice_parameter_names(model: str) -> tuple[str, ...]:
    """Expose behavioral parameter meanings separately from inferred environmental noise parameters."""
    if model == "constant":
        return ("bet_minus_logit", "observe_logit")
    if model == "leaky_threshold":
        return (
            "evidence_decay",
            "initial_threshold",
            "final_threshold",
            "threshold_onset",
            "threshold_noise",
        )
    names = ("reward_sensitivity", "observe_bias", "outcome_color_bias")
    if model == "bayes_time":
        names += ("remaining_horizon_weight",)
    elif model in ("ats_binary_parameter_ig", "bayes_state_ig"):
        names += ("information_weight",)
    return names


def threshold_probabilities(
    parameters: np.ndarray, blocks: list[ObserveBlock]
) -> np.ndarray:
    """Implement the released five-parameter leaky-threshold likelihood with ML rather than its Bayesian prior."""
    decay, initial, final, onset, noise = parameters
    feedback = np.stack([block.feedback for block in blocks]).astype(float)
    evidence_updated = lfilter([1.0], [1.0, -(1 - decay)], feedback, axis=1)
    evidence = np.column_stack((np.zeros(len(blocks)), evidence_updated[:, :-1]))
    time = np.arange(1, 51)
    fraction = np.maximum(0, (time - 50 * onset) / (50 * (1 - onset)))
    threshold = initial + fraction * (final - initial)
    low = ndtr((-evidence - threshold) / noise)
    high = ndtr((evidence - threshold) / noise)
    middle = np.maximum(0, 1 - low - high)
    probability = np.stack((low, middle, high), axis=-1).reshape(-1, 3)
    probability /= probability.sum(axis=1, keepdims=True)
    return (1 - LAPSE) * probability + LAPSE / 3


def fit_choice_model(
    model: str, blocks: list[ObserveBlock], features: list[np.ndarray] | None = None
) -> ChoiceFit:
    """Fit one participant using only the supplied training blocks, leaving test sessions untouched."""
    choices = np.concatenate([block.choices for block in blocks])
    valid = choices != -2
    indices = choices[valid] + 1
    if model == "leaky_threshold":
        bounds = ((0.0001, 0.9999), (0.01, 12), (0.01, 12), (0, 0.9), (0.05, 10))
        starts = ((0.05, 3, 0.5, 0.5, 1), (0.2, 1, 1, 0.2, 1), (0.4, 5, 1, 0.7, 2))

        def objective(parameters):
            """Score only nonmissed choices while retaining missed trials in the history."""
            probability = threshold_probabilities(parameters, blocks)[valid]
            return float(-np.log(probability[np.arange(len(indices)), indices]).sum())

        results = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-10, "maxiter": 500},
            )
            for start in starts
        ]
        best = select_optimization(results, objective, bounds)
        boundary = tuple(
            i
            for i, (value, (low, high)) in enumerate(zip(best.x, bounds))
            if min(value - low, high - value) < 1e-5 * max(1, high - low)
        )
        return ChoiceFit(
            model, best.x, float(best.fun), bool(best.success), boundary, len(starts)
        )
    if features is None:
        features = [block_features(block) for block in blocks]
    designs = [choice_design(feature, model) for feature in features]
    design = np.concatenate([item[0] for item in designs])[valid]
    return fit_design(model, design, indices, designs[0][1])


def predict_choice_model(
    fit: ChoiceFit, blocks: list[ObserveBlock], features: list[np.ndarray] | None = None
) -> np.ndarray:
    """Predict every decision before its feedback with frozen participant parameters."""
    if fit.model == "leaky_threshold":
        return threshold_probabilities(fit.parameters, blocks)
    if features is None:
        features = [block_features(block) for block in blocks]
    design = np.concatenate(
        [choice_design(feature, fit.model)[0] for feature in features]
    )
    return categorical_probabilities(design, fit.parameters)


def simulate_observe_participant(
    model: str,
    parameters: np.ndarray,
    seed: int,
    train_blocks: int = 6,
    test_blocks: int = 5,
) -> list[ObserveBlock]:
    """Generate new closed-loop feedback under the simulated choices, never reusing a human feedback trace."""
    rng = np.random.default_rng(seed)
    blocks = []
    for block_index in range(train_blocks + test_blocks):
        hidden = int(rng.integers(2))
        belief = BinaryNoiseBelief()
        choices = np.full(50, -2, dtype=int)
        feedback = np.zeros(50, dtype=int)
        features = np.zeros((50, 5))
        session = "train" if block_index < train_blocks else "test"
        for trial in range(50):
            features[trial] = belief.features()
            if model == "leaky_threshold":
                temporary = ObserveBlock(
                    "synthetic", session, str(block_index), choices, feedback
                )
                probability = threshold_probabilities(parameters, [temporary])[trial]
            else:
                design, _ = choice_design(features, model)
                probability = categorical_probabilities(
                    design[trial : trial + 1], parameters
                )[0]
            choices[trial] = rng.choice(ACTIONS, p=probability)
            outcome = (1 if hidden else -1) * (1 if rng.random() < 0.8 else -1)
            if choices[trial] == 0:
                feedback[trial] = outcome
            belief.advance(int(feedback[trial]))
            if rng.random() < 0.05:
                hidden = 1 - hidden
        blocks.append(
            ObserveBlock(
                "synthetic", session, str(block_index), choices.copy(), feedback.copy()
            )
        )
    return blocks


def interval_design(information_gains: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Original ATS softmax uses identifiable gain and delay coefficients; constant observation cost cancels."""
    gains = np.asarray(information_gains, dtype=float)
    intervals = np.asarray(intervals, dtype=float)
    if gains.ndim != 2 or gains.shape[1] != len(intervals) or np.any(intervals <= 0):
        raise ValueError("Invalid interval design.")
    return np.stack((gains, -np.broadcast_to(intervals, gains.shape)), axis=2)
