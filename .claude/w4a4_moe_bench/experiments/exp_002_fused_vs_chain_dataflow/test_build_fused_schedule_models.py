from __future__ import annotations

import unittest

from build_fused_schedule_models import build_models


def row(m: int, phase: str, **metrics):
    return {
        "m": m,
        "arm": "cutedsl_bf16_fused" if phase == "fused_main" else "cutlass_bf16_chain",
        "phase": phase,
        "metrics": metrics,
    }


class ScheduleModelTest(unittest.TestCase):
    def test_exact_reduction_and_local_traffic_context(self) -> None:
        physical_rows = 32768
        fp4_ops = physical_rows * 3 * 2 * 2048 * 512
        evidence = {
            "targets": [
                row(
                    256,
                    "fused_main",
                    fp4_to_fp32_tensor_ops=fp4_ops,
                    local_load_sectors=1998848,
                    local_store_sectors=1998848,
                    local_total_footprint_bytes=174945536,
                    global_reduction_sectors=1048576,
                ),
                row(256, "fc1", local_total_footprint_bytes=22426624),
                row(256, "fc2", local_total_footprint_bytes=41746688),
            ]
        }
        # The production builder expects both fixed M cases; duplicate the same
        # synthetic topology under M=8192 with its exact logical reduction count.
        evidence["targets"].extend(
            [
                row(
                    8192,
                    "fused_main",
                    fp4_to_fp32_tensor_ops=81152 * 3 * 2 * 2048 * 512,
                    local_load_sectors=4950272,
                    local_store_sectors=4950272,
                    local_total_footprint_bytes=491943168,
                    global_reduction_sectors=33554432,
                ),
                row(8192, "fc1", local_total_footprint_bytes=53295616),
                row(8192, "fc2", local_total_footprint_bytes=101164736),
            ]
        )
        static_local_sass = {
            "static_instruction_facts": {
                "stack_roundtrip_model": {
                    "total_stored_32bit_words_per_lane": 122,
                    "stl64_covered_32bit_words": 108,
                    "all_stored_slots_have_a_later_static_ldl": True,
                }
            }
        }
        models = build_models(evidence, static_local_sass)
        small = models["cases"][0]
        self.assertEqual(small["compute_task_count"], 1024)
        self.assertEqual(
            small["local_traffic_context"][
                "measured_chain_fc1_fc2_local_footprint_bytes"
            ],
            64173312,
        )
        self.assertEqual(
            small["local_traffic_context"]["source_phase_attribution"],
            "program-order inference; no source lineinfo",
        )
        self.assertEqual(small["local_traffic_context"]["store_sector_residual"], 0)
        self.assertEqual(small["partial_output_reduction_model"]["residual_sectors"], 0)


if __name__ == "__main__":
    unittest.main()
