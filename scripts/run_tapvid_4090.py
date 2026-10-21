"""Run the fixed RTX 4090 ATS experiment; compare with Track-On-R's published row."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats_vision.server_experiment import (
    PROJECT,
    build_report,
    gpu_preflight,
    load_profile,
    prepare_server_data,
    run_evaluation,
    run_training,
)


def main() -> None:
    """Keep downloading, fitting, frozen evaluation, and reporting explicit and resumable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "plan",
            "prepare",
            "preflight",
            "train",
            "validate",
            "test",
            "report",
            "all",
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/tapvid_4090.json"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.config, args.output_root, args.seeds, args.max_hours)
    if args.stage == "plan":
        print(json.dumps(profile, indent=2))
        print(
            "Published competitor only: Track-On-R; no external fitting/checkpoint required."
        )
        return
    if args.stage in {"prepare", "all"}:
        print(prepare_server_data(profile), flush=True)
    if args.stage == "preflight":
        print(gpu_preflight(profile), flush=True)
    if args.stage in {"train", "all"}:
        if not run_training(profile, args.resume):
            sys.exit(2)
    if args.stage in {"validate", "all"}:
        run_evaluation(profile, "validation", args.resume)
    if args.stage in {"test", "all"}:
        run_evaluation(profile, "test", args.resume)
    if args.stage in {"report", "all"}:
        print(build_report(profile), flush=True)


if __name__ == "__main__":
    main()
