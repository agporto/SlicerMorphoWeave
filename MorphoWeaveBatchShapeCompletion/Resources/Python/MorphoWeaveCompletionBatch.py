"""Filesystem orchestration for completion; deliberately independent of Slicer.

The processor receives one specimen and an empty staging directory. It must
raise on any fitting/export error. A specimen becomes resumable only after all
its files and its success manifest have been committed atomically.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Mapping, Optional

MODEL_EXTENSIONS = (".ply", ".vtp", ".vtk", ".stl", ".obj")
MARKUP_EXTENSIONS = (".mrk.json", ".fcsv", ".json")
MANIFEST_NAME = "completion_batch.json"
SCHEMA_VERSION = 1
SUMMARY_FIELDS = (
    "specimen", "input_mesh", "input_landmarks", "output_directory", "status",
    "elapsed_seconds", "settings_sha256", "error",
)


@dataclass(frozen=True)
class BatchSpecimen:
    name: str
    mesh: Path
    landmarks: Optional[Path] = None


def _stem(path: Path, extensions) -> str:
    for extension in extensions:
        if path.name.lower().endswith(extension):
            return path.name[:-len(extension)]
    raise ValueError(f"Unsupported file: {path}")


def _index_files(directory: Path, extensions):
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    index = {}
    for path in sorted(directory.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
        if not path.is_file() or not path.name.lower().endswith(extensions):
            continue
        stem = _stem(path, extensions)
        if stem in ("", ".", ".."):
            raise ValueError(f"Unsafe specimen filename: {path.name}")
        key = stem.casefold()
        if key in index:
            raise ValueError(
                f"Ambiguous specimen name '{stem}': {index[key].name} and {path.name}"
            )
        index[key] = path.absolute()
    return index


def discover_specimens(input_directory, landmark_directory=None):
    """Discover a flat mesh directory and pair markups by case-insensitive stem.

    Missing landmarks are represented by None, allowing the runner to report a
    per-specimen failure in landmark mode without dropping other specimens.
    Ambiguous filenames are rejected before any fitting or output creation.
    """
    meshes = _index_files(Path(input_directory).expanduser(), MODEL_EXTENSIONS)
    if not meshes:
        raise ValueError("The input directory contains no supported mesh files")
    markups = (
        _index_files(Path(landmark_directory).expanduser(), MARKUP_EXTENSIONS)
        if landmark_directory else {}
    )
    return [
        BatchSpecimen(_stem(path, MODEL_EXTENSIONS), path, markups.get(key))
        for key, path in meshes.items()
    ]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _input_identity(path):
    if path is None:
        return None
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def _atomic_write(path, write):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_summary(path, rows):
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write(path, write)


@contextmanager
def _directory_lock(directory):
    lock = directory / ".morphoweave_batch.lock"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"Another batch may be using {directory}. If Slicer previously crashed, "
            f"verify that no batch is running, then remove {lock.name} and resume."
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _artifact_records(directory):
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Output artifacts must not be symbolic links: {path}")
        if not path.is_file():
            continue
        if path.name == MANIFEST_NAME:
            raise ValueError(f"Processor wrote the reserved filename {MANIFEST_NAME}")
        size = path.stat().st_size
        if size == 0:
            raise IOError(f"Empty output artifact: {path.name}")
        records.append({
            "path": path.relative_to(directory).as_posix(),
            "size": size,
            "sha256": file_sha256(path),
        })
    if not records:
        raise IOError("Completion produced no output files")
    return records


def _resumable(directory, signature):
    try:
        if directory.is_symlink():
            return False
        with open(directory / MANIFEST_NAME, encoding="utf-8") as stream:
            manifest = json.load(stream)
        if (manifest.get("schema_version") != SCHEMA_VERSION
                or manifest.get("status") != "success"
                or manifest.get("signature") != signature
                or not manifest.get("artifacts")):
            return False
        root = directory.resolve()
        for artifact in manifest["artifacts"]:
            relative = Path(artifact["path"])
            if relative.is_absolute() or ".." in relative.parts:
                return False
            path = directory / relative
            if path.is_symlink() or root not in path.resolve().parents:
                return False
            if (not path.is_file() or path.stat().st_size != artifact["size"]
                    or file_sha256(path) != artifact["sha256"]):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def run_batch(
    specimens,
    output_directory,
    processor: Callable[[BatchSpecimen, Path], None],
    *,
    context: Mapping,
    require_landmarks=False,
    resume=True,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[dict, int, int], None]] = None,
):
    """Run sequentially, checkpointing a CSV after each specimen.

    Existing specimen directories are never overwritten. Resume skips only
    successful outputs whose inputs, context, and artifact hashes still match.
    Cancellation is cooperative between specimens, not inside native fitting.
    A failed processor never publishes a partial specimen directory.
    """
    specimens = list(specimens)
    if not specimens:
        raise ValueError("No specimens to process")
    names = set()
    output = Path(output_directory).expanduser().resolve()
    for specimen in specimens:
        name = specimen.name
        if (not name or name in (".", "..") or "/" in name or "\\" in name
                or name.casefold() in names):
            raise ValueError(f"Invalid or duplicate specimen name: {name!r}")
        names.add(name.casefold())
        if output == specimen.mesh.resolve().parent:
            raise ValueError("Input and output directories must be different")
    # Freeze context so callers cannot mutate it during progress callbacks.
    context = json.loads(_json_bytes(dict(context)))
    settings_hash = hashlib.sha256(_json_bytes(context)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    rows = [{
        "specimen": specimen.name,
        "input_mesh": str(specimen.mesh),
        "input_landmarks": str(specimen.landmarks) if specimen.landmarks else "",
        "output_directory": str(output / specimen.name),
        "status": "pending", "elapsed_seconds": "",
        "settings_sha256": settings_hash, "error": "",
    } for specimen in specimens]
    summary = output / "batch_summary.csv"
    with _directory_lock(output):
        _write_summary(summary, rows)
        for index, (specimen, row) in enumerate(zip(specimens, rows)):
            if should_cancel is not None and should_cancel():
                for remaining in rows[index:]:
                    remaining["status"] = "cancelled"
                _write_summary(summary, rows)
                break
            started = time.perf_counter()
            row["status"] = "running"
            _write_summary(summary, rows)
            if progress is not None:
                progress(dict(row), index, len(rows))
            # A progress callback may deliver a cancellation event before fitting.
            if should_cancel is not None and should_cancel():
                for remaining in rows[index:]:
                    remaining["status"] = "cancelled"
                _write_summary(summary, rows)
                break
            temporary = None
            try:
                if require_landmarks and specimen.landmarks is None:
                    raise ValueError("Landmark-assisted mode requires a matching landmark file")
                identity = {
                    "schema_version": SCHEMA_VERSION, "settings_sha256": settings_hash,
                    "mesh": _input_identity(specimen.mesh),
                    "landmarks": _input_identity(specimen.landmarks),
                    "require_landmarks": bool(require_landmarks),
                }
                signature = hashlib.sha256(_json_bytes(identity)).hexdigest()
                destination = output / specimen.name
                if resume and _resumable(destination, signature):
                    row["status"] = "skipped"
                else:
                    if destination.exists() or destination.is_symlink():
                        raise FileExistsError(
                            "Existing outputs do not match a verified resumable completion "
                            "(or resume is disabled). Choose a new output directory; "
                            "existing results were not overwritten."
                        )
                    temporary = Path(tempfile.mkdtemp(prefix=f".{specimen.name}.tmp-", dir=str(output)))
                    processor(specimen, temporary)
                    # Detect input edits during fitting rather than blessing a stale result.
                    if (identity["mesh"] != _input_identity(specimen.mesh)
                            or identity["landmarks"] != _input_identity(specimen.landmarks)):
                        raise RuntimeError("An input file changed while the specimen was being processed")
                    artifacts = _artifact_records(temporary)
                    manifest = {
                        "schema_version": SCHEMA_VERSION, "status": "success",
                        "signature": signature, "specimen": specimen.name,
                        "inputs": identity, "context": context, "artifacts": artifacts,
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    _atomic_write(temporary / MANIFEST_NAME,
                                  lambda stream: json.dump(manifest, stream, indent=2, sort_keys=True))
                    if destination.exists() or destination.is_symlink():
                        raise FileExistsError(f"Output directory appeared during fitting: {destination}")
                    temporary.rename(destination)
                    temporary = None
                    row["status"] = "success"
            except Exception as error:
                row["status"] = "failed"
                row["error"] = f"{type(error).__name__}: {error}"
                logging.exception("Batch shape completion failed for %s", specimen.name)
            finally:
                if temporary is not None:
                    shutil.rmtree(temporary)
            row["elapsed_seconds"] = f"{time.perf_counter() - started:.6f}"
            _write_summary(summary, rows)
            if progress is not None:
                progress(dict(row), index + 1, len(rows))
    return rows
