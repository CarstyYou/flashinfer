#!/usr/bin/env python3
"""Extract provenance-locked cubin/PTX/SASS/resource evidence for exp_006."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def one_artifact(capture: dict[str, Any], suffix: str) -> dict[str, Any]:
    matches = [
        item for item in capture["jit_artifacts"] if item["path"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} artifact, found {len(matches)}")
    return matches[0]


def run(command: Sequence[str]) -> bytes:
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {list(command)!r}\n"
            + completed.stderr.decode(errors="replace")
        )
    return completed.stdout


def tool_identity(name: str) -> dict[str, str]:
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(name)
    path = Path(resolved).resolve()
    version = run([str(path), "--version"]).decode(errors="replace").strip()
    if not version:
        raise ValueError(f"{path} returned an empty version")
    return {"requested": name, "resolved_path": str(path), "sha256": sha256(path), "version": version}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvdisasm", default="nvdisasm")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()

    capture_path = args.capture.resolve()
    jit_root = args.jit_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable static output exists: {output}")
    capture = read_json(capture_path)
    cubin_entry = one_artifact(capture, ".cubin")
    ptx_entry = one_artifact(capture, ".ptx")
    cubin = jit_root / cubin_entry["path"]
    ptx = jit_root / ptx_entry["path"]
    for path, entry in ((cubin, cubin_entry), (ptx, ptx_entry)):
        if not path.is_file() or path.stat().st_size != int(entry["size"]):
            raise ValueError(f"retained JIT artifact missing or size drifted: {path}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"retained JIT artifact hash drifted: {path}")

    nvdisasm = tool_identity(args.nvdisasm)
    cuobjdump = tool_identity(args.cuobjdump)
    commands = {
        "sass": [nvdisasm["resolved_path"], "-c", str(cubin)],
        "resource": [cuobjdump["resolved_path"], "--dump-resource-usage", str(cubin)],
        "elf": [cuobjdump["resolved_path"], "--dump-elf", str(cubin)],
    }
    outputs = {name: run(command) for name, command in commands.items()}
    if b"MoEDynamicKernel" not in outputs["sass"]:
        raise ValueError("selected cubin does not contain MoEDynamicKernel")

    temporary = output.with_name(f".{output.name}.in-progress.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copyfile(cubin, temporary / "kernel.cubin")
        shutil.copyfile(ptx, temporary / "kernel.ptx")
        (temporary / "kernel.sass").write_bytes(outputs["sass"])
        (temporary / "resource.txt").write_bytes(outputs["resource"])
        (temporary / "elf.txt").write_bytes(outputs["elf"])
        artifact_hashes = {
            name: sha256(temporary / filename)
            for name, filename in {
                "cubin": "kernel.cubin",
                "ptx": "kernel.ptx",
                "sass": "kernel.sass",
                "resource": "resource.txt",
                "elf": "elf.txt",
            }.items()
        }
        provenance = {
            "schema": "exp006.static-provenance.v1",
            "arm": capture["arm"],
            "capture_json": {"path": str(capture_path), "sha256": sha256(capture_path)},
            "container_image_digest": capture["runtime"]["image_digest"],
            "provider": {
                "jit_root": str(jit_root),
                "cubin": {
                    "path": str(cubin),
                    "relative_path": cubin_entry["path"],
                    "sha256": cubin_entry["sha256"],
                    "size": cubin_entry["size"],
                },
                "ptx": {
                    "path": str(ptx),
                    "relative_path": ptx_entry["path"],
                    "sha256": ptx_entry["sha256"],
                    "size": ptx_entry["size"],
                },
            },
            "tools": {"nvdisasm": nvdisasm, "cuobjdump": cuobjdump},
            "commands": commands,
            "artifacts": artifact_hashes,
        }
        (temporary / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "arm": capture["arm"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
