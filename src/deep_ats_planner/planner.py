"""DQN-style planner over task actions and ATS sampling intervals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class JointAction:
    task_action: int
    dt: int


@dataclass(frozen=True)
class ReplayTransition:
    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray
    done: bool
    duration: int = 1


def bellman_target(
    rewards, next_values, dones, durations, gamma: float, discount_basis: str
):
    """Use physical durations or an explicitly chosen per-decision objective."""
    if discount_basis not in {"physical_time", "decision"}:
        raise ValueError("Unknown discount basis.")
    exponents = (
        durations if discount_basis == "physical_time" else torch.ones_like(durations)
    )
    return rewards + gamma**exponents * (1 - dones) * next_values


class ReplayBuffer:
    """Fixed-capacity replay storage for value-function training."""

    def __init__(self, capacity: int):
        self.transitions: deque[ReplayTransition] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: ReplayTransition) -> None:
        self.transitions.append(transition)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> list[ReplayTransition]:
        indices = rng.choice(len(self.transitions), size=int(batch_size), replace=False)
        return [self.transitions[int(index)] for index in indices]


class QNetwork(nn.Module):
    """Small MLP that approximates Q(z, action, dt) for all joint actions."""

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(output_dim)),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class DeepATSPlanner:
    """Belief-conditioned value learner that jointly chooses action and dt."""

    def __init__(
        self,
        input_dim: int,
        task_action_count: int,
        action_space: tuple[int, ...],
        hidden_size: int,
        learning_rate: float,
        gamma: float,
        replay_capacity: int,
        device: str = "cpu",
        discount_basis: str = "physical_time",
    ):
        self.device = torch.device(device)
        self.joint_actions = tuple(
            JointAction(task_action=task_action, dt=int(dt))
            for task_action in range(int(task_action_count))
            for dt in action_space
        )
        self.gamma = float(gamma)
        self.discount_basis = discount_basis
        self.network = QNetwork(input_dim, len(self.joint_actions), hidden_size).to(
            self.device
        )
        self.target_network = QNetwork(
            input_dim, len(self.joint_actions), hidden_size
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=float(learning_rate)
        )
        self.replay = ReplayBuffer(replay_capacity)
        self.loss_fn = nn.SmoothL1Loss()

    def select(
        self,
        state: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
    ) -> tuple[int, int, int, np.ndarray]:
        if rng.random() < float(epsilon):
            action_index = int(rng.integers(0, len(self.joint_actions)))
            q_values = self.q_values(state)
        else:
            q_values = self.q_values(state)
            action_index = int(np.argmax(q_values))
        action = self.joint_actions[action_index]
        return action.task_action, action.dt, action_index, q_values

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(
                state[None, :], dtype=torch.float32, device=self.device
            )
            values = self.network(tensor).cpu().numpy()[0]
        return values.astype(float)

    def train_step(self, batch_size: int, rng: np.random.Generator) -> float | None:
        if len(self.replay) < int(batch_size):
            return None
        batch = self.replay.sample(batch_size, rng)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        action_indices = torch.as_tensor(
            [item.action_index for item in batch],
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        )
        durations = torch.as_tensor(
            [item.duration for item in batch], dtype=torch.float32, device=self.device
        )

        predicted = self.network(states).gather(1, action_indices[:, None]).squeeze(1)
        with torch.no_grad():
            next_values = self.target_network(next_states).max(dim=1).values
            target = bellman_target(
                rewards, next_values, dones, durations, self.gamma, self.discount_basis
            )

        loss = self.loss_fn(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def update_target(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
