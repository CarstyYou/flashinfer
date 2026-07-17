#!/usr/bin/env python3
"""Close the attribution-only arm without inventing an IR-to-SASS mapping."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from exp003_common import DEFAULT_RESULTS, MAIN_BUNDLE_WORDS, read_json, write_json


def compiler_artifacts(preparation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in preparation.get("jit_artifacts", [])
        if Path(str(item.get("path", ""))).suffix in {".mlir", ".ptx", ".cubin"}
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    results = args.results.resolve()
    static = read_json(results / "static_spill_evidence.json")
    ncu = read_json(results / "ncu" / "spill_evidence.json")
    arm = "up_first_attribution"
    baseline = static["arms"]["baseline"]
    swapped = static["arms"][arm]
    delta = ncu["deltas"][arm]
    base_prep = read_json(results / "arms" / "baseline" / "preparation.json")
    swap_prep = read_json(results / "arms" / arm / "preparation.json")
    correctness = read_json(results / "correctness" / f"{arm}.json")
    checks = {
        "semantic_order_overlay_is_distinct": (
            base_prep["runtime"]["source"]["overlay_sha256"]
            != swap_prep["runtime"]["source"]["overlay_sha256"]
        ),
        "main_108_word_bundle_preserved": (
            baseline["local"]["stored_words_by_opcode_width"].get("STL.64")
            == MAIN_BUNDLE_WORDS
            and swapped["local"]["stored_words_by_opcode_width"].get("STL.64")
            == MAIN_BUNDLE_WORDS
        ),
        "non_local_opcode_projection_preserved": static["deltas"][arm][
            "selected_opcode_projection_equal_except_local"
        ],
        "tail_dynamic_delta_closed": delta["dynamic_14_word_closure_pass"],
        "tensor_work_identity": delta["work_identity_pass"],
        "quant_aware_oracle_pass": correctness["quant_aware_oracle_pass"],
        "strict_cross_arm_correctness_pass": correctness["gate_pass"],
        # The retained compiler dumps have no source lineinfo or allocation map
        # that connects Cute/MLIR SSA values to ptxas physical spill registers.
        "ir_ptx_sass_physical_def_use_bridge": False,
    }
    payload = {
        "schema": "exp003.spill-root-cause.attribution-evidence.v1",
        "arm": arm,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "formal_verdict": "source_and_program_order_inference_only",
        "reason": (
            "static and dynamic evidence preserve the 108-word bundle and exact work, "
            "but no compiler artifact bridges semantic accumulator SSA to the physical "
            "STL/LDL register chain; strict cross-arm correctness is also invalid because "
            "baseline replay self-drift exceeds the pre-registered hard cap"
        ),
        "observed_tail_effect": {
            "stack_bytes_per_thread": [
                baseline["resource"]["stack_bytes_per_thread"],
                swapped["resource"]["stack_bytes_per_thread"],
            ],
            "stored_words_per_lane": [
                baseline["local"]["stored_words"],
                swapped["local"]["stored_words"],
            ],
            "local_sector_reduction_per_direction": delta[
                "local_load_sector_reduction"
            ],
            "executed_local_instruction_reduction_per_direction": delta[
                "executed_local_load_instruction_reduction"
            ],
        },
        "compiler_artifacts": {
            "baseline": compiler_artifacts(base_prep),
            arm: compiler_artifacts(swap_prep),
        },
    }
    write_json(results / "attribution_evidence.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
