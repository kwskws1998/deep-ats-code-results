"""Create a dedicated Linux/Python 3.12 CUDA environment for the ATS-only experiment.

Run with the server's Python 3.12. No activation, shell script, cognitive/RL
dependencies, competitor package, or competitor checkpoint is required.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import venv


def setup_commands(python: Path, project: Path) -> list[list[str]]:
    """Keep the CUDA wheel index separate from the ordinary pinned dependencies."""
    return [
        [str(python), "-m", "pip", "install", "pip==25.2"],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "torch==2.12.1+cu126",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(project / "requirements/tapvid-4090.txt"),
        ],
        [str(python), "-m", "pip", "check"],
    ]


def main() -> None:
    """Install into an isolated environment and record its resolved dependency set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path(".venv4090"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    environment = args.venv.resolve()
    python = environment / "bin/python"
    commands = setup_commands(python, project)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "python_required": "3.12",
                    "platform": "Linux x86_64",
                    "venv": str(environment),
                    "commands": commands,
                },
                indent=2,
            )
        )
        return
    if sys.version_info[:2] != (3, 12):
        parser.error("Run this installer with Python 3.12, for example python3.12.")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        parser.error("The RTX 4090 environment targets Linux x86_64.")
    if environment.exists() and not (environment / "pyvenv.cfg").is_file():
        parser.error("Refusing to use a non-venv directory as the environment.")
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run(
        [str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 12)"],
        check=True,
    )
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=project, check=True)
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import json, torch; assert torch.cuda.is_available(), 'CUDA unavailable: check the NVIDIA driver and GPU assignment'; p = torch.cuda.get_device_properties(0); print(json.dumps(dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=p.name, total_vram_bytes=p.total_memory)))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    )
    (environment / "resolved-requirements.txt").write_text(freeze.stdout)
    (environment / "cuda-environment.json").write_text(probe.stdout)
    print(probe.stdout.strip())
    print(f"Ready: {python} scripts/run_tapvid_4090.py all", flush=True)


if __name__ == "__main__":
    main()
