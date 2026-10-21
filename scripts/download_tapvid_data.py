"""Download official TAP-Vid evaluation data or selected CoTracker3 Kubric scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

TAPNET_REVISION = "730cda1c730877cfedbe01bf87fb1cadb78a565d"
DAVIS_URL = "https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip"
DAVIS_MD5 = "a7e786a1a5d26ba941b7c10bed21c20e"


def fetch_json(url: str):
    """Read public repository metadata without credentials."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    """Hash large source archives without loading them into memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_entries(revision: str) -> dict:
    """Follow Hugging Face pagination so high scene indices remain downloadable."""
    url = f"https://huggingface.co/api/datasets/facebook/CoTracker3_Kubric/tree/{revision}?recursive=false&limit=1000"
    entries = {}
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            entries.update({item["path"]: item for item in json.load(response)})
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return entries


def download(url: str, destination: Path, expected: str, algorithm: str) -> dict:
    """Resume a verified byte range and atomically publish only a matching file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest_file(destination, algorithm) != expected:
            raise ValueError(f"Existing file checksum mismatch: {destination}")
    else:
        partial = destination.with_suffix(destination.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and (
                response.status != 206
                or not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                )
            ):
                raise RuntimeError(
                    "Server did not honor resume range; remove only the .part file to restart."
                )
            last = time.monotonic()
            with partial.open("ab" if offset else "wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
                    offset += len(chunk)
                    if time.monotonic() - last > 10:
                        print(f"{destination.name}: {offset / 1e6:.1f} MB", flush=True)
                        last = time.monotonic()
        if digest_file(partial, algorithm) != expected:
            raise ValueError(f"Downloaded checksum mismatch: {partial}")
        partial.replace(destination)
    return dict(
        url=url,
        path=str(destination.resolve()),
        bytes=destination.stat().st_size,
        sha256=digest_file(destination),
        published_checksum=expected,
        checksum_algorithm=algorithm,
    )


def main() -> None:
    """Require explicit scene selection; never silently download the whole corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/external/tapvid"))
    parser.add_argument("--davis", action="store_true")
    parser.add_argument("--kubric-scenes", nargs="*", type=int, default=[])
    parser.add_argument(
        "--kubric-revision",
        help="Freeze a published dataset revision before downloading.",
    )
    args = parser.parse_args()
    if not args.davis and not args.kubric_scenes:
        parser.error("Select --davis and/or explicit --kubric-scenes indices.")
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "downloads.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.davis:
        manifest["davis"] = download(
            DAVIS_URL, args.root / "tapvid_davis.zip", DAVIS_MD5, "md5"
        )
    if args.kubric_scenes:
        if any(index < 0 for index in args.kubric_scenes):
            parser.error("Scene indices must be nonnegative.")
        metadata = fetch_json(
            "https://huggingface.co/api/datasets/facebook/CoTracker3_Kubric"
        )
        revision = args.kubric_revision or manifest.get(
            "kubric_revision", metadata["sha"]
        )
        if manifest.get("kubric_revision", revision) != revision:
            raise ValueError(
                "Existing download manifest uses a different Kubric revision."
            )
        entries = repository_entries(revision)
        manifest["kubric_revision"] = revision
        for index in sorted(set(args.kubric_scenes)):
            name = f"{index:04d}.tar.gz"
            if name not in entries:
                raise ValueError(
                    f"Scene {name} is absent from the pinned repository revision."
                )
            checksum = entries[name]["lfs"]["oid"]
            url = f"https://huggingface.co/datasets/facebook/CoTracker3_Kubric/resolve/{revision}/{name}?download=true"
            manifest[f"kubric_{index:04d}"] = download(
                url, args.root / "kubric" / name, checksum, "sha256"
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    source_url = f"https://raw.githubusercontent.com/google-deepmind/tapnet/{TAPNET_REVISION}/tapnet/tapvid/evaluation_datasets.py"
    source_path = args.root / "official_evaluation_datasets.py"
    with urllib.request.urlopen(source_url, timeout=60) as response:
        source = response.read()
    source_path.write_bytes(source)
    manifest["official_evaluator"] = dict(
        url=source_url,
        revision=TAPNET_REVISION,
        sha256=hashlib.sha256(source).hexdigest(),
        path=str(source_path.resolve()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified sources: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
