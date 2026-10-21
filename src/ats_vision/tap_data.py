"""TAP point/occlusion data, safe NumPy decoding, and source-level split preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
import zipfile

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    """Hash a source or prepared clip without reading the whole file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArrayUnpickler(pickle.Unpickler):
    """Decode legacy published NumPy containers without arbitrary global imports."""

    def find_class(self, module, name):
        """Allow only NumPy array construction and byte encoding used by archives."""
        import codecs

        core = np._core if hasattr(np, "_core") else np.core
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("_codecs", "encode"): codecs.encode,
        }
        for prefix in ("numpy.core", "numpy._core"):
            allowed[(prefix + ".multiarray", "_reconstruct")] = (
                core.multiarray._reconstruct
            )
            allowed[(prefix + ".multiarray", "scalar")] = core.multiarray.scalar
            allowed[(prefix + ".numeric", "_frombuffer")] = core.numeric._frombuffer
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Disallowed data global: {module}.{name}")
        return allowed[(module, name)]


def load_numpy_dictionary(stream) -> dict:
    """Read the object NPY format used by released Kubric coordinate annotations."""
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    if not dtype.hasobject or shape != () or fortran:
        raise ValueError("Expected a scalar NumPy dictionary annotation.")
    value = ArrayUnpickler(stream).load().item()
    if not isinstance(value, dict):
        raise ValueError("Expected dictionary annotations.")
    return value


@dataclass
class TapClip:
    """RGB video T,H,W,3; tracks N,T,xy in image-fraction raster coordinates."""

    frames: np.ndarray
    points: np.ndarray
    occluded: np.ndarray
    source_id: str
    dataset: str

    def validate(self) -> None:
        """Validate visible annotations and preserve out-of-frame occluded positions."""
        if (
            self.frames.ndim != 4
            or self.frames.shape[-1] != 3
            or self.frames.dtype != np.uint8
        ):
            raise ValueError("frames must be T,H,W,3 uint8 RGB.")
        if self.frames.shape[0] < 2 or min(self.frames.shape[1:3]) < 8:
            raise ValueError("A clip needs >=2 frames and image dimensions >=8.")
        n, length = self.points.shape[:2]
        if self.points.shape != (n, self.frames.shape[0], 2) or n < 1:
            raise ValueError("points must be nonempty N,T,2.")
        if self.points.dtype != np.float32 or not np.isfinite(self.points).all():
            raise ValueError("points must be finite float32 values.")
        if self.occluded.shape != (n, length) or self.occluded.dtype != np.bool_:
            raise ValueError("occluded must be boolean N,T; True means invisible.")
        if np.any(
            (self.points[~self.occluded] < 0) | (self.points[~self.occluded] > 1)
        ):
            raise ValueError("Visible point coordinates must be in [0,1].")
        if not np.any(~self.occluded) or not self.source_id or not self.dataset:
            raise ValueError(
                "Need visible queries and explicit source/dataset identities."
            )

    def queries(self) -> tuple[np.ndarray, np.ndarray]:
        """Use the official first-visible [t,y,x] convention, skipping never-visible tracks."""
        indices = np.flatnonzero((~self.occluded).any(1))
        clocks = (~self.occluded[indices]).argmax(1)
        xy = self.points[indices, clocks]
        return indices, np.column_stack((clocks, xy[:, 1], xy[:, 0])).astype(np.float32)

    def save(self, path: Path) -> None:
        """Save pickle-free arrays and primitive source metadata."""
        self.validate()
        np.savez_compressed(
            path,
            frames=self.frames,
            points=self.points,
            occluded=self.occluded,
            source_id=np.asarray(self.source_id),
            dataset=np.asarray(self.dataset),
        )


def load_tap_clip(path: Path) -> TapClip:
    """Read a prepared clip with object deserialization disabled."""
    with np.load(path, allow_pickle=False) as arrays:
        clip = TapClip(
            arrays["frames"],
            arrays["points"],
            arrays["occluded"],
            str(arrays["source_id"]),
            str(arrays["dataset"]),
        )
    clip.validate()
    return clip


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize RGB raster images without changing fractional coordinate labels."""
    if size < 8:
        raise ValueError("image_size must be >=8.")
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR)
            )
            for frame in frames
        ]
    )


def read_kubric_archive(path: Path, image_size: int) -> TapClip:
    """Read CoTracker3's published coords/visibility format without extracting paths.

    Despite its name, the source visibility array means occlusion: the official
    KubricMovifDataset parser applies logical_not before treating it as visible.
    """
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        expected_name = path.name.split(".")[0] + ".npy"
        annotations = [
            item
            for item in members
            if item.isfile() and Path(item.name).name == expected_name
        ]
        images = sorted(
            [
                item
                for item in members
                if item.isfile()
                and "/frames/" in item.name
                and item.name.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
            key=lambda item: item.name,
        )
        if len(annotations) != 1 or not images:
            raise ValueError(
                "Expected one coordinate NPY and a frames directory per Kubric archive."
            )
        with archive.extractfile(annotations[0]) as stream:
            labels = load_numpy_dictionary(stream)
        decoded = []
        original_size = None
        for item in images:
            with archive.extractfile(item) as stream, Image.open(stream) as image:
                image = image.convert("RGB")
                if original_size is not None and image.size != original_size:
                    raise ValueError("Inconsistent frame dimensions.")
                original_size = image.size
                decoded.append(
                    np.asarray(
                        image.resize(
                            (image_size, image_size), Image.Resampling.BILINEAR
                        )
                    )
                )
    points = np.asarray(labels["coords"], dtype=np.float32)[..., :2].copy()
    occluded = np.asarray(labels["visibility"], dtype=bool).copy()
    if points.shape[:2] != (occluded.shape[0], len(decoded)) or occluded.shape[
        1
    ] != len(decoded):
        raise ValueError("Kubric annotations must use track,frame axes.")
    points /= np.asarray(original_size, dtype=np.float32)
    invalid = ~np.isfinite(points).all(-1) | (points < 0).any(-1) | (points > 1).any(-1)
    occluded |= invalid
    points = np.nan_to_num(points, nan=0, posinf=0, neginf=0)
    clip = TapClip(
        np.stack(decoded),
        points,
        occluded,
        f"cotracker3_kubric:{path.name.split('.')[0]}",
        "cotracker3_kubric",
    )
    clip.validate()
    return clip


def read_davis_archive(path: Path, image_size: int):
    """Yield official DAVIS clips from its checksum-verified NumPy pickle container."""
    with zipfile.ZipFile(path) as archive:
        members = [
            name for name in archive.namelist() if name.endswith((".pkl", ".pickle"))
        ]
        if len(members) != 1:
            raise ValueError("Expected exactly one official TAP-Vid-DAVIS pickle.")
        with archive.open(members[0]) as stream:
            contents = ArrayUnpickler(stream).load()
    if not isinstance(contents, dict):
        raise ValueError("Expected video-name keyed DAVIS annotations.")
    for name in sorted(contents):
        item = contents[name]
        frames = item["video"]
        if not isinstance(frames, np.ndarray) or frames.ndim != 4:
            frames = np.stack(
                [
                    np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
                    for value in frames
                ]
            )
        clip = TapClip(
            resize_frames(frames, image_size),
            np.asarray(item["points"], np.float32),
            np.asarray(item["occluded"], bool),
            f"tapvid_davis:{name}",
            "tapvid_davis",
        )
        clip.validate()
        yield str(name), clip
        del contents[name]


def load_tap_manifest(path: Path, verify: bool = True) -> dict:
    """Validate source-level isolation and forbid DAVIS in fitting/selection roles."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported TAP manifest schema.")
    source_roles, ids, fingerprints = {}, set(), {}
    for entry in manifest["clips"]:
        if entry["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split role.")
        if entry["dataset"] != "cotracker3_kubric" and entry["split"] != "test":
            raise ValueError("Only Kubric scenes may enter training or validation.")
        previous = source_roles.setdefault(entry["source_id"], entry["split"])
        if previous != entry["split"] or entry["clip_id"] in ids:
            raise ValueError("Source video or clip leaks between dataset roles.")
        ids.add(entry["clip_id"])
        role = fingerprints.setdefault(entry["frame_sha256"], entry["split"])
        if role != entry["split"]:
            raise ValueError("Exact frames reused across dataset roles.")
        destination = (path.parent / entry["path"]).resolve()
        if not destination.is_relative_to(path.parent):
            raise ValueError("Prepared clip path escapes its dataset root.")
        if verify and sha256(destination) != entry["sha256"]:
            raise ValueError(f"Prepared clip changed: {destination}")
    manifest["root"] = str(path.parent)
    return manifest


def prepare_tap_data(
    kubric_paths: list[Path],
    davis_path: Path | None,
    output: Path,
    image_size: int = 256,
    train_length: int = 24,
    validation_fraction: float = 0.1,
    seed: int = 17,
) -> Path:
    """Split whole Kubric scenes before windowing; retain complete DAVIS test clips."""
    if len(kubric_paths) < 2 or not 0 < validation_fraction < 1 or train_length < 2:
        raise ValueError(
            "Need >=2 Kubric scenes, a proper validation fraction, and length >=2."
        )
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prepared output must be a fresh directory.")
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(map(Path, kubric_paths))
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Duplicate Kubric archive names.")
    rng = np.random.default_rng(seed)
    validation_count = min(
        len(paths) - 1, max(1, int(np.ceil(len(paths) * validation_fraction)))
    )
    validation_names = {
        paths[index].name for index in rng.permutation(len(paths))[:validation_count]
    }
    entries, sources = [], {}

    def publish(clip: TapClip, identifier: str, split: str) -> None:
        """Record immutable clip bytes and a source-level split assignment."""
        destination = output / f"{identifier}.npz"
        clip.save(destination)
        entries.append(
            dict(
                clip_id=identifier,
                path=destination.name,
                split=split,
                source_id=clip.source_id,
                dataset=clip.dataset,
                sha256=sha256(destination),
                frame_sha256=hashlib.sha256(clip.frames.tobytes()).hexdigest(),
                frames=len(clip.frames),
                tracks=len(clip.points),
            )
        )

    for path in paths:
        clip = read_kubric_archive(path, image_size)
        sources[str(path.resolve())] = sha256(path)
        split = "validation" if path.name in validation_names else "train"
        if split == "validation":
            publish(clip, f"kubric_{path.name.split('.')[0]}_full", split)
            continue
        starts = range(0, len(clip.frames) - train_length + 1, train_length)
        published = 0
        for start in starts:
            window = TapClip(
                clip.frames[start : start + train_length],
                clip.points[:, start : start + train_length],
                clip.occluded[:, start : start + train_length],
                clip.source_id,
                clip.dataset,
            )
            if not np.any(~window.occluded[:, :-1]):
                continue
            publish(window, f"kubric_{path.name.split('.')[0]}_{start:05d}", split)
            published += 1
        if not published:
            raise ValueError(f"No valid windows from {path}.")
    if davis_path is not None:
        sources[str(Path(davis_path).resolve())] = sha256(davis_path)
        for index, (name, clip) in enumerate(
            read_davis_archive(davis_path, image_size)
        ):
            publish(clip, f"davis_{index:03d}", "test")
    manifest = dict(
        schema_version=1,
        seed=seed,
        image_size=image_size,
        train_length=train_length,
        query_mode="first",
        training_source="published CoTracker3 Kubric variant; custom scene-level validation split",
        sources=sources,
        clips=entries,
    )
    if davis_path is not None:
        digest = hashlib.md5()
        with Path(davis_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest["davis_archive_md5"] = digest.hexdigest()
        manifest["official_davis_source"] = (
            digest.hexdigest() == "a7e786a1a5d26ba941b7c10bed21c20e"
        )
    destination = output / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    load_tap_manifest(destination)
    return destination
