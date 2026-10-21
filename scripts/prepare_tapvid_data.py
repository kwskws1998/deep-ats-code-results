"""Prepare source-disjoint Kubric train/validation and complete TAP-Vid test clips."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats_vision.tap_data import prepare_tap_data


def main():
    """Keep dataset roles and resolution explicit in a frozen preparation manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubric-root", type=Path, required=True)
    parser.add_argument("--davis", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-length", type=int, default=24)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = prepare_tap_data(
        sorted(args.kubric_root.glob("*.tar.gz")),
        args.davis,
        args.output_root,
        args.image_size,
        args.train_length,
        args.validation_fraction,
        args.seed,
    )
    print(output)


if __name__ == "__main__":
    main()
