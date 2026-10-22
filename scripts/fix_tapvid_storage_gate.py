"""Remove the original fixed storage gate in an already uploaded server checkout.

Run from the server's project directory with Python. No packages or network are
needed. The original source is backed up before the first modification.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_repo(project: Path) -> bool:
    """Patch the known original guard once and validate Python before writing."""
    path = Path(project) / "src/ats_vision/server_experiment.py"
    source = path.read_text()
    function = source.index("def prepare_server_data(")
    begin = source.index("    free = shutil.disk_usage(raw).free / 2**30\n", function)
    end = source.index('    start, stop = profile["scene_indices"]\n', begin)
    block = source[begin:end]
    if "required_free" not in block:
        print("The fixed storage gate is already absent.")
        return False
    if "raise RuntimeError(" not in block or "minimum_prepare_free_gib" not in block:
        raise ValueError("Unrecognized storage guard; no source was modified.")
    replacement = (
        "    free = shutil.disk_usage(raw).free / 2**30\n"
        '    print(f"Available storage at {raw}: {free:.1f} GiB (informational).", flush=True)\n'
    )
    patched = source[:begin] + replacement + source[end:]
    compile(patched, str(path), "exec")
    backup = path.with_suffix(path.suffix + ".before_storage_fix")
    if backup.exists():
        raise FileExistsError(f"Existing backup preserved: {backup}")
    backup.write_text(source)
    path.write_text(patched)
    print(f"Removed the fixed storage gate: {path}")
    return True


def main() -> None:
    """Apply the one-file fix without reinstalling the experiment environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args()
    patch_repo(args.project)


if __name__ == "__main__":
    main()
