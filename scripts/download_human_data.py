"""Download versioned public behavioral datasets and retain verifiable source archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


SOURCES = {
    "stovol_learning_2024": {
        "paper": "https://doi.org/10.1038/s41467-024-53459-z",
        "repository": "https://zenodo.org/records/13840905",
        "files": [
            (
                "data.zip",
                "https://zenodo.org/records/13840905/files/data.zip?download=1",
                "59cdae0b66f5868ae9268729fa863d0e",
            ),
            (
                "code.zip",
                "https://zenodo.org/records/13840905/files/code.zip?download=1",
                "12f30e337f40c4f6052cf06ac6303362",
            ),
        ],
    },
    "observe_bet_2017": {
        "paper": "https://doi.org/10.3758/s13415-017-0556-2",
        "repository": "https://github.com/TommyBlanchard/ObserveBet",
        "commit": "9a0ae02937ddfc9dabaf24cd4155dc9ed112d0ae",
        "files": [
            (
                "repository.zip",
                "https://codeload.github.com/TommyBlanchard/ObserveBet/zip/9a0ae02937ddfc9dabaf24cd4155dc9ed112d0ae",
                None,
            ),
        ],
    },
}


def download_dataset(name: str, root: Path) -> None:
    """Verify fixed archive checksums when provided and reject paths outside the extraction root."""
    source = SOURCES[name]
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    previous_path = directory / "source_manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    known_hashes = {
        record["file"]: record["sha256"] for record in previous.get("archives", [])
    }
    records = []
    for filename, url, expected_md5 in source["files"]:
        path = directory / filename
        if not path.exists():
            request = urllib.request.Request(
                url, headers={"User-Agent": "ATS-cognitive-validation/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(100_000_001)
            if len(payload) > 100_000_000:
                raise ValueError("Archive exceeds the declared download size limit.")
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(payload)
            temporary.replace(path)
        payload = path.read_bytes()
        if (
            filename in known_hashes
            and hashlib.sha256(payload).hexdigest() != known_hashes[filename]
        ):
            raise ValueError(f"Previously recorded source archive changed: {path}")
        md5 = hashlib.md5(payload).hexdigest()
        if expected_md5 is not None and md5 != expected_md5:
            raise ValueError(f"Published checksum mismatch: {path}")
        destination = (directory / "unpacked").resolve()
        with zipfile.ZipFile(path) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 200_000_000:
                raise ValueError("Archive exceeds the declared extraction size limit.")
            for entry in archive.infolist():
                if (
                    not (destination / entry.filename)
                    .resolve()
                    .is_relative_to(destination)
                ):
                    raise ValueError("Archive contains an unsafe path.")
            archive.extractall(destination)
        records.append(
            {
                "file": filename,
                "url": url,
                "bytes": len(payload),
                "md5": md5,
                "published_md5": expected_md5,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        print(f"verified {path}", flush=True)
    metadata = {key: value for key, value in source.items() if key != "files"}
    metadata.update(
        archives=records,
        purpose="Local analysis of public human behavior; source code is inspected, not executed.",
    )
    (directory / "source_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main() -> None:
    """Fetch one or both public datasets without executing downloaded analysis programs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--dataset", choices=tuple(SOURCES), nargs="+", default=list(SOURCES)
    )
    args = parser.parse_args()
    for name in args.dataset:
        download_dataset(name, args.root)


if __name__ == "__main__":
    main()
