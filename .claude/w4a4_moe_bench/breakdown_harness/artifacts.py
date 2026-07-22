"""Content identity and atomic serialization for experiment-owned artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def source_manifest(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Bind logical harness components to their exact source contents."""
    files: dict[str, dict[str, Any]] = {}
    fingerprint_input: dict[str, str] = {}
    for name, raw_path in sorted(paths.items()):
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing harness source {name}: {path}")
        sha256 = file_sha256(path)
        files[name] = {
            "path": str(path),
            "sha256": sha256,
            "size_bytes": path.stat().st_size,
        }
        fingerprint_input[name] = sha256
    return {
        "files": files,
        "fingerprint_sha256": canonical_sha256(fingerprint_input),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("CSV rows have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    suffixes = {
        ".so",
        ".cu",
        ".cuh",
        ".cpp",
        ".ptx",
        ".cubin",
        ".sass",
        ".json",
        ".mlir",
        ".ncu-rep",
        ".nsys-rep",
        ".csv",
    }
    if not root.exists():
        return []
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]
