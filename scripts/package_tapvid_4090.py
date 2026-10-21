"""Build a portable code-only server bundle, excluding datasets and previous results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    """Include reproducibility sources and hashes without copying local caches or data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/tapvid_4090_code.zip")
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    paths = [project / name for name in ("README.md", "pyproject.toml", ".gitignore")]
    for name in (
        "src",
        "scripts",
        "tests",
        "docs",
        "configs",
        "requirements",
        "references",
    ):
        paths.extend(
            path
            for path in (project / name).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".md", ".json", ".txt", ".toml"}
        )
    paths = sorted(set(paths))
    hashes = {
        str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(
            "Choose a new bundle name; existing bundles are preserved."
        )
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(project))
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(hashes, indent=2) + "\n")
    print(
        f"{args.output.resolve()} ({args.output.stat().st_size} bytes, {len(paths)} files)"
    )


if __name__ == "__main__":
    main()
