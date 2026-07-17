#!/usr/bin/env python3
"""Build the compact, fail-closed exp_004 early-stop evidence.

The normal/control preparations are complete.  The probe preparation stopped
at its timing-event gate, so this collector accepts that explicit failure
record, extracts all three retained cubins, and records only facts needed to
close the experiment without publishing phase shares.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from build_binary_identity import (
    EXPECTED_BASELINE_CUBIN_SHA256,
    EXPECTED_BASELINE_SASS_SHA256,
    _locked_artifacts,
    _select_target_cubin,
    analyze_arm,
)
from exp004_common import (
    ALL_ARMS,
    DEFAULT_RESULTS,
    EXPECTED_CUTLASS_COMMIT,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_FLASHINFER_COMMIT,
    EXPECTED_KERNEL_SHA256,
    EXPECTED_TASK_TAIL,
    EXPECTED_WRAPPER_SHA256,
    MEASUREMENT_CONTROL,
    NORMAL,
    PROBE,
    SENTINEL,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)
from run_exp004 import preparation_path


TOOLCHAIN_FIELDS = (
    "nvcc",
    "ptxas",
    "python",
    "torch",
    "cuda_runtime",
    "image_digest",
    "python_deps_sha256",
)
GPU_FIELDS = (
    "uuid",
    "name",
    "pci_bus_id",
    "compute_capability",
    "sm_count",
)
SEMANTIC_FIELDS = ("omma", "utmaldg", "ldsm", "bar", "atomg", "redg", "ldg")


def _single_ptx(preparation: Mapping[str, Any]) -> Path:
    candidates = _locked_artifacts(preparation, ".ptx")
    matches = [
        path for path in candidates if "clock64" in path.read_text(errors="replace")
    ]
    if len(matches) == 1:
        return matches[0]
    if len(candidates) != 1:
        raise ValueError(f"expected one target PTX, got {len(candidates)}")
    return candidates[0]


def _artifact_hashes(
    analysis: Mapping[str, Any],
    ptx: Path,
    preparation: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "cubin": str(analysis["identity"]["cubin_sha256"]),
        "sass": str(analysis["identity"]["sass_sha256"]),
        "ptx": file_sha256(ptx),
        "resource_dump": str(analysis["raw_outputs"]["resource_sha256"]),
        "elf_dump": str(analysis["raw_outputs"]["elf_sha256"]),
        "jit_artifact_set": str(preparation["jit_artifact_set_sha256"]),
    }


def _compact_arm(
    analysis: Mapping[str, Any],
    ptx: Path,
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    local_counts = Counter(
        {
            str(opcode): int(count)
            for opcode, count in analysis["spill"]["local_opcode_counts"].items()
        }
    )
    projection = analysis["selected_opcode_projection"]
    return {
        "cubin_sha256": analysis["identity"]["cubin_sha256"],
        "sass_sha256": analysis["identity"]["sass_sha256"],
        "resource": analysis["resource"],
        "local_sass": {
            "stl_instruction_count": sum(
                count
                for opcode, count in local_counts.items()
                if opcode.startswith("STL")
            ),
            "ldl_instruction_count": sum(
                count
                for opcode, count in local_counts.items()
                if opcode.startswith("LDL")
            ),
            "spill_annotation_count": analysis["spill"]["compiler_annotation_count"],
        },
        "semantic_projection": {
            field: int(projection[field]) for field in SEMANTIC_FIELDS
        },
        "artifact_hashes": _artifact_hashes(analysis, ptx, preparation),
    }


def _same_identity(
    preparations: Mapping[str, Mapping[str, Any]],
    *,
    section: str,
    fields: Sequence[str],
) -> bool:
    return all(
        len(
            {
                canonical_sha256(preparations[arm]["runtime"][section][field])
                for arm in ALL_ARMS
            }
        )
        == 1
        for field in fields
    )


def build(
    *,
    results: Path,
    probe_failure_path: Path,
    cuobjdump: str,
    nvdisasm: str,
) -> dict[str, Any]:
    results = results.resolve()
    preparations: dict[str, Mapping[str, Any]] = {
        NORMAL: read_json(preparation_path(results, NORMAL)),
        MEASUREMENT_CONTROL: read_json(preparation_path(results, MEASUREMENT_CONTROL)),
        PROBE: read_json(probe_failure_path.resolve()),
    }
    if (
        preparations[NORMAL].get("status") != "complete"
        or preparations[MEASUREMENT_CONTROL].get("status") != "complete"
    ):
        raise ValueError("normal and measurement-control preparations must be complete")
    failure = preparations[PROBE]
    if failure.get("status") != "failed_timing_event_gate":
        raise ValueError("probe failure must be an explicit timing-event-gate failure")

    analyses: dict[str, Mapping[str, Any]] = {}
    ptx_paths: dict[str, Path] = {}
    for arm in ALL_ARMS:
        preparation = preparations[arm]
        cubin = _select_target_cubin(
            preparation, explicit=None, nvdisasm=nvdisasm, label=f"{arm} cubin"
        )
        analyses[arm] = analyze_arm(
            arm=arm,
            preparation=preparation,
            cubin=cubin,
            cuobjdump=cuobjdump,
            nvdisasm=nvdisasm,
            raw_root=results / "raw" / "blocked_binary" / arm,
        )
        ptx_paths[arm] = _single_ptx(preparation)
        write_json(
            results / "raw" / "blocked_binary" / arm / "parsed.json",
            analyses[arm],
        )

    normal_runtime = preparations[NORMAL]["runtime"]
    production = normal_runtime["source"]["production"]
    source_gate = all(
        preparation["runtime"]["source"]["locked_source_commit"]
        == EXPECTED_FLASHINFER_COMMIT
        and preparation["runtime"]["source"]["cutlass_commit"]
        == EXPECTED_CUTLASS_COMMIT
        and preparation["runtime"]["source"]["production"]["kernel"]["sha256"]
        == EXPECTED_KERNEL_SHA256
        and preparation["runtime"]["source"]["production"]["dispatch"]["sha256"]
        == EXPECTED_DISPATCH_SHA256
        and preparation["runtime"]["source"]["production"]["wrapper"]["sha256"]
        == EXPECTED_WRAPPER_SHA256
        for preparation in preparations.values()
    )
    hardware_gate = _same_identity(
        preparations, section="gpu", fields=GPU_FIELDS
    ) and all(
        bool(preparation["runtime"]["hardware_gate"]["gate_pass"])
        for preparation in preparations.values()
    )
    toolchain_gate = all(
        len({str(preparations[arm]["runtime"][field]) for arm in ALL_ARMS}) == 1
        for field in TOOLCHAIN_FIELDS
    )

    failed_replay = int(failure["failed_replay"])
    failed_timing_path = probe_failure_path.parent / f"timing_{failed_replay}.pt"
    timing = torch.load(failed_timing_path, map_location="cpu", weights_only=True)
    observed_ticks = int(torch.count_nonzero(timing["timing_ticks"] != SENTINEL))
    observed_cta = int(torch.count_nonzero(timing["task_cta_z"] != SENTINEL))
    event_gate = failure["failure"]["gate"]
    if observed_ticks != int(event_gate["observed_tick_writes"]):
        raise ValueError("failed timing tensor disagrees with event gate")
    failed_output = failure["outputs"][failed_replay]
    failed_workspace = failure["workspace_gates"][failed_replay]

    probe_ptx = ptx_paths[PROBE].read_text(errors="replace")
    compact_arms = {
        arm: _compact_arm(analyses[arm], ptx_paths[arm], preparations[arm])
        for arm in ALL_ARMS
    }
    if compact_arms[NORMAL]["cubin_sha256"] != EXPECTED_BASELINE_CUBIN_SHA256:
        raise ValueError("production cubin anchor drift")
    if compact_arms[NORMAL]["sass_sha256"] != EXPECTED_BASELINE_SASS_SHA256:
        raise ValueError("production SASS anchor drift")

    payload: dict[str, Any] = {
        "schema": "exp004.blocked-preflight.v1",
        "status": "blocked_by_measurement_gate",
        "identity_gates": {
            "source": source_gate,
            "hardware": hardware_gate,
            "toolchain": toolchain_gate,
        },
        "evidence_identity": {
            "source": {
                "flashinfer_commit": normal_runtime["source"]["locked_source_commit"],
                "cutlass_commit": normal_runtime["source"]["cutlass_commit"],
                "kernel_sha256": production["kernel"]["sha256"],
                "dispatch_sha256": production["dispatch"]["sha256"],
                "wrapper_sha256": production["wrapper"]["sha256"],
            },
            "gpu": {
                **{field: normal_runtime["gpu"][field] for field in GPU_FIELDS},
                "driver": normal_runtime["gpu"]["driver"],
            },
            "toolchain": {
                **{field: normal_runtime[field] for field in TOOLCHAIN_FIELDS},
                "cutlass_dsl_module": normal_runtime["imports"]["cutlass_python"],
                "cutlass_dsl_version": normal_runtime["imports"][
                    "cutlass_python_version"
                ],
            },
        },
        "runtime_observations": {
            arm: {
                "applications_graphics_clock_mhz": preparations[arm]["runtime"]["gpu"][
                    "applications_graphics_clock_mhz"
                ],
                "graphics_clock_mhz": preparations[arm]["runtime"]["gpu"][
                    "graphics_clock_mhz"
                ],
                "max_graphics_clock_mhz": preparations[arm]["runtime"]["gpu"][
                    "max_graphics_clock_mhz"
                ],
                "power_draw_w": preparations[arm]["runtime"]["gpu"]["power_draw_w"],
                "lease_id": preparations[arm]["runtime"]["lease_id"],
                "jit_artifact_set_sha256": preparations[arm]["jit_artifact_set_sha256"],
            }
            for arm in ALL_ARMS
        },
        "arms": compact_arms,
        "event_contract": {
            "expected_tick_writes": int(event_gate["expected_tick_writes"]),
            "observed_tick_writes": observed_ticks,
            "expected_task_cta_writes": EXPECTED_TASK_TAIL,
            "observed_task_cta_writes": observed_cta,
        },
        "probe_lowering": {
            "ptx_clock64_count": probe_ptx.count("clock64"),
            "ptx_probe_store_count": probe_ptx.count("st.global.u64"),
        },
        "probe_preparation_gates": {
            "reference_correctness": bool(failed_output["gate"]["gate_pass"]),
            "output_contract": bool(failed_output["output_contract"]["gate_pass"]),
            "workspace_contract": bool(failed_workspace["gate_pass"]),
            "reference_metrics": {
                field: float(failed_output[field])
                for field in (
                    "cosine",
                    "relative_l2",
                    "max_abs",
                    "token_rel_l2_p99",
                )
            },
        },
        "failed_probe_preparation": {
            "path": str(probe_failure_path.resolve()),
            "sha256": file_sha256(probe_failure_path),
            "timing_tensor_sha256": file_sha256(failed_timing_path),
            "runtime_zero_write_cause": "unresolved",
        },
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--probe-failure", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--nvdisasm", default="nvdisasm")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(
        results=args.results,
        probe_failure_path=args.probe_failure,
        cuobjdump=args.cuobjdump,
        nvdisasm=args.nvdisasm,
    )
    output = args.results.resolve() / "raw" / "blocked_preflight.json"
    write_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
