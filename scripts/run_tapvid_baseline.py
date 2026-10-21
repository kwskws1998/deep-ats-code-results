"""Run observed-frame-only LK or an optional official TAPNext++ checkpoint."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats_vision.tap_baselines import evaluate_baseline


def main():
    """Require explicit pretrained weights and label sparse-cadence adaptation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("lk", "tapnextpp"), default="lk")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=("uniform", "random"),
        default=["uniform", "random"],
    )
    parser.add_argument(
        "--extrapolation", choices=("hold", "velocity"), default="velocity"
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    print(evaluate_baseline(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
