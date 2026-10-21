"""Verify a validation suite and every child run against recorded artifact hashes."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.artifacts import verify_run


def main():
    """Emit a machine-readable integrity report and fail if any completed run has drifted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check-current-source", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifests = sorted(args.root.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError("No run manifests found.")
    reports = {
        str(path.parent): verify_run(path.parent, args.check_current_source)
        for path in manifests
    }
    report = {
        "runs": len(reports),
        "passed": all(not failures for failures in reports.values()),
        "failures": reports,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        if args.report.resolve().is_relative_to(args.root.resolve()):
            raise ValueError(
                "Save the report outside the checked suite so its artifact inventory stays unchanged."
            )
        args.report.write_text(text)
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
