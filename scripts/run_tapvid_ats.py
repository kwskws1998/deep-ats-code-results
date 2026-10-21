"""Train/validate on Kubric, then explicitly evaluate a frozen model on TAP-Vid."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats_vision.tap_training import TAPTrainConfig, train_tap, evaluate_tap


def main():
    """Separate fitting from final testing and expose subset/compute limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    evaluate = commands.add_parser("evaluate")
    for command in (train, evaluate):
        command.add_argument("--data-manifest", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--threads", type=int, default=2)
        command.add_argument("--seed", type=int, default=17)
    train.add_argument("--resume", action="store_true")
    for name, default in (
        ("steps", 1000),
        ("queries-per-step", 4),
        ("validation-every", 100),
        ("validation-queries", 16),
        ("validation-clips", 16),
        ("feature-dim", 32),
        ("hidden-size", 48),
        ("max-gap", 16),
        ("warmup-steps", 0),
        ("data-cache-clips", 0),
    ):
        train.add_argument(f"--{name}", type=int, default=default)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--max-hours", type=float, default=12)
    train.add_argument("--validation-observation-fraction", type=float, default=0.5)
    train.add_argument(
        "--learning-rate-schedule", choices=("constant", "cosine"), default="constant"
    )
    train.add_argument("--minimum-learning-rate", type=float, default=0.0)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument(
        "--fractions", nargs="+", type=float, default=[0.25, 0.5, 1.0]
    )
    evaluate.add_argument(
        "--policies",
        nargs="+",
        default=[
            "uniform",
            "random",
            "total_uncertainty",
            "ats",
            "ats_fixed_q",
            "ats_fixed_r",
        ],
    )
    evaluate.add_argument("--max-queries", type=int, default=0)
    evaluate.add_argument("--max-clips", type=int, default=0)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "train":
        resume = args.pop("resume")
        output = train_tap(TAPTrainConfig(**args), resume)
    else:
        args["output"] = args.pop("output_root")
        output = evaluate_tap(**args)
    print(output)


if __name__ == "__main__":
    main()
