"""Run a bounded neural video ATS pilot or train on labeled NPZ splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats_vision.experiment import VisionConfig, run_experiment


def main() -> None:
    """Expose compute and dataset settings without implicit long-running jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/vision_ats_pilot")
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for option, default in (
        ("seed", 17),
        ("train-clips", 128),
        ("validation-clips", 24),
        ("test-clips", 32),
        ("sequence-length", 24),
        ("image-size", 24),
        ("observation-budget", 8),
        ("hidden-size", 32),
        ("warmup-epochs", 4),
        ("epochs", 12),
        ("batch-size", 16),
        ("max-gap", 8),
        ("threads", 2),
    ):
        parser.add_argument(f"--{option}", type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    output = run_experiment(VisionConfig(**vars(args)))
    print(f"Saved frozen-model comparison to {output}")


if __name__ == "__main__":
    main()
