"""Diagnose frozen/live checkpoints or run bounded train-only probes in a separate checkout."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats_vision.tap_diagnostics import diagnose_checkpoint
from ats_vision.tap_probes import benchmark_queries, overfit_probe
from ats_vision.tap_failure_audit import failure_audit


def main():
    """Keep final-test labels unavailable to development diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checkpoint = commands.add_parser("checkpoint")
    overfit = commands.add_parser("overfit")
    benchmark = commands.add_parser("benchmark")
    history = commands.add_parser("history")
    audit = commands.add_parser("failure-audit")
    history.add_argument("--training-csv", type=Path, required=True)
    for command in (checkpoint, overfit, benchmark, audit):
        command.add_argument("--data-manifest", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--threads", type=int, default=2)
        command.add_argument("--seed", type=int, default=17)
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    checkpoint.add_argument("--max-clips", type=int, default=2)
    checkpoint.add_argument("--max-queries", type=int, default=4)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--max-clips", type=int, default=2)
    audit.add_argument("--max-queries", type=int, default=8)
    audit.add_argument("--gradient-frames", type=int, default=16)
    audit.add_argument(
        "--include-lk", action=argparse.BooleanOptionalAction, default=True
    )
    for command in (overfit, benchmark):
        command.add_argument("--max-frames", type=int, default=16)
        command.add_argument("--feature-dim", type=int, default=16)
        command.add_argument("--hidden-size", type=int, default=24)
    overfit.add_argument("--steps", type=int, default=200)
    overfit.add_argument("--learning-rate", type=float, default=0.003)
    benchmark.add_argument("--batch-size", type=int, default=4)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--encoder-chunk-size", type=int, default=8)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "history":
        with args["training_csv"].open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("Training history has no validation rows.")
        fields = [
            key
            for key in rows[0]
            if key
            in {
                "step",
                "loss",
                "learning_rate",
                "visible_position_mse",
                "average_jaccard",
                "average_pts_within_thresh",
                "occlusion_accuracy",
            }
        ]
        print(" | ".join(fields))
        for row in rows[-10:]:
            print(" | ".join(row[key] for key in fields))
        print(
            "AJ/within-threshold/OA are fractions (multiply by 100 for percent); MSE is mean squared error per coordinate in 256-pixel units."
        )
        return
    args["output"] = args.pop("output_root")
    action = {
        "checkpoint": diagnose_checkpoint,
        "overfit": overfit_probe,
        "benchmark": benchmark_queries,
        "failure-audit": failure_audit,
    }[command]
    print(action(**args))


if __name__ == "__main__":
    main()
