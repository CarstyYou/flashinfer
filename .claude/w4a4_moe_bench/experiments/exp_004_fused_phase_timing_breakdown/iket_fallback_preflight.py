#!/usr/bin/env python3
"""Record the exp_004 coarse-IKET fallback admission status without executing it."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from exp004_common import DEFAULT_RESULTS, file_sha256, write_json


AUDITED_IKET_VERSION = "0.7.10"
DEFAULT_KDK_HELPER = Path(
    "/home/scratch.xiy_gpu/kernel-dev-kit/breakdown/iket_safe_capture.py"
)


def _distribution_version(run_iket: Path) -> str | None:
    root = run_iket.parent.parent
    for pattern in (
        "lib/python*/site-packages/iket-*.dist-info/METADATA",
        "lib64/python*/site-packages/iket-*.dist-info/METADATA",
    ):
        for metadata in sorted(root.glob(pattern)):
            name = version = None
            for line in metadata.read_text(errors="replace").splitlines():
                if line.startswith("Name:"):
                    name = line.partition(":")[2].strip()
                elif line.startswith("Version:"):
                    version = line.partition(":")[2].strip()
            if name and name.lower() == "iket" and version:
                return version
    return None


def preflight(run_iket_value: str, helper: Path) -> dict[str, object]:
    helper = helper.resolve()
    if not helper.is_file():
        raise FileNotFoundError(f"KDK IKET safety helper is missing: {helper}")
    resolved = shutil.which(run_iket_value)
    run_iket = Path(resolved).resolve() if resolved else None
    version = _distribution_version(run_iket) if run_iket else None
    provider_ready = bool(run_iket and version == AUDITED_IKET_VERSION)
    return {
        "schema": "exp004.iket-fallback-preflight.v1",
        "status": (
            "provider_ready_but_not_admitted"
            if provider_ready
            else "blocked_provider_unavailable_or_version_drift"
        ),
        "admission": (
            "not admitted: primary sparse clock probe has not failed its "
            "plumbing/capture-feasibility gate"
        ),
        "kdk_safe_capture": {
            "path": str(helper),
            "sha256": file_sha256(helper),
        },
        "provider": {
            "requested": run_iket_value,
            "resolved": str(run_iket) if run_iket else None,
            "observed_version": version,
            "required_audited_version": AUDITED_IKET_VERSION,
            "ready": provider_ready,
        },
        "coarse_overlay": {
            "status": "not_built_until_fallback_is_admitted",
            "required_ranges": [
                "task_envelope",
                "fc1_gate",
                "fc1_up",
                "swiglu_q1",
                "fc2_gemm",
                "fc2_epilogue_scatter",
                "gate_tma",
                "gate_pass_wait",
                "up_tma",
                "down_tma",
                "final_pass_wait",
            ],
            "forbidden": "old per-OMMA/per-S2R/per-wait cadence overlay",
        },
        "required_if_admitted": [
            "IKET-compiler normal/no-marker identity",
            "IKET-compiler coarse-marker candidate identity",
            "fresh KDK iket_safe_capture.py output",
            "same resource/spill/SASS/work/correctness gates as primary",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-iket", default="run-iket")
    parser.add_argument("--kdk-helper", type=Path, default=DEFAULT_KDK_HELPER)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "iket_fallback_preflight.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = preflight(args.run_iket, args.kdk_helper)
    write_json(args.output.resolve(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
