#!/usr/bin/env python3
"""CPU-only gates for the exp_016 Route/Q0 overlay builder."""

from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "build_overlays.py"
SPEC = importlib.util.spec_from_file_location("exp016_build_overlays", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
overlays = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlays)


class BuildOverlaysTest(unittest.TestCase):
    def test_build_locks_identity_mechanism_and_work_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "w4a4_moe_bench"
            source = root / "moe_dynamic_kernel_opt.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(overlays.SOURCE.read_bytes())
            output = root / "experiments/exp_016/results/overlays"
            identity = overlays.build_overlays(
                source=source, overlay_root=output, memory_root=root
            )

            baseline = (
                output / overlays.BASELINE_NAME / "moe_dynamic_kernel.py"
            ).read_bytes()
            candidate = (
                output / overlays.CANDIDATE_NAME / "moe_dynamic_kernel.py"
            ).read_bytes()
            expected_baseline, expected_candidate, source_role = overlays.resolve_arms(
                source.read_text(encoding="utf-8")
            )
            self.assertEqual(baseline, expected_baseline.encode())
            self.assertEqual(candidate, expected_candidate.encode())
            self.assertEqual(identity["source_role"], source_role)
            selected = baseline if source_role == overlays.BASELINE_NAME else candidate
            self.assertEqual(selected, source.read_bytes())
            self.assertNotEqual(candidate, baseline)
            ast.parse(candidate.decode("utf-8"))
            self.assertTrue(identity["mechanism_gate"]["gate_pass"])
            self.assertTrue(all(identity["mechanism_gate"]["checks"].values()))
            self.assertEqual(
                identity,
                json.loads((output / "identity.json").read_text(encoding="utf-8")),
            )
            self.assertEqual(
                identity["counter_units"][overlays.CANDIDATE_NAME],
                {"unit": "token", "claim_per_cta": 9},
            )
            ledger = identity["work_ledger_m8192_topk8_h2048"]
            self.assertEqual(ledger["logical_routes"], 65_536)
            self.assertEqual(ledger["quant_blocks_candidate"], 8_388_608)
            self.assertEqual(ledger["bf16_block_loads_baseline"], 8_388_608)
            self.assertEqual(ledger["bf16_block_loads_candidate"], 1_048_576)
            self.assertEqual(ledger["productive_claims_baseline"], 3_641)
            self.assertEqual(ledger["productive_claims_candidate"], 911)
            self.assertEqual(
                ledger["packed_fp4_stores_baseline"],
                ledger["packed_fp4_stores_candidate"],
            )

    def test_candidate_has_token_major_loop_and_per_expert_scale(self) -> None:
        source = overlays.SOURCE.read_text(encoding="utf-8")
        _, candidate, _ = overlays.resolve_arms(source)
        self.assertIn("token_idx = batch_base + warp_idx", candidate)
        self.assertIn(
            "route_gs = cute.make_rmem_tensor((8,), cutlass.Float32)", candidate
        )
        self.assertIn("for cache_slot in cutlass.range_constexpr(8):", candidate)
        self.assertIn("gs_value = route_gs[cache_slot]", candidate)
        self.assertNotIn("route_scale_slot", candidate)
        self.assertNotIn(
            "claim_count = producer_batch_pairs",
            candidate,
        )

    def test_exact_anchors_reject_missing_and_duplicate_sources(self) -> None:
        source, _, _ = overlays.resolve_arms(
            overlays.SOURCE.read_text(encoding="utf-8")
        )
        for malformed in ("", source + source):
            with self.subTest(size=len(malformed)), self.assertRaises(RuntimeError):
                overlays.apply_candidate(malformed)

    def test_promoted_candidate_round_trips_to_locked_baseline(self) -> None:
        baseline, candidate, _ = overlays.resolve_arms(
            overlays.SOURCE.read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "w4a4_moe_bench"
            source = root / "moe_dynamic_kernel_opt.py"
            source.parent.mkdir(parents=True)
            source.write_text(candidate, encoding="utf-8")
            output = root / "experiments/exp_016/results/overlays"
            identity = overlays.build_overlays(
                source=source, overlay_root=output, memory_root=root
            )
            self.assertEqual(identity["source_role"], overlays.CANDIDATE_NAME)
            self.assertEqual(
                (output / overlays.BASELINE_NAME / "moe_dynamic_kernel.py").read_text(),
                baseline,
            )
            self.assertEqual(
                (
                    output / overlays.CANDIDATE_NAME / "moe_dynamic_kernel.py"
                ).read_text(),
                candidate,
            )

    def test_source_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "w4a4_moe_bench"
            source = root / "moe_dynamic_kernel_opt.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(overlays.SOURCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "locked opt source drift"):
                overlays.build_overlays(
                    source=source,
                    overlay_root=root / "results/overlays",
                    memory_root=root,
                )

    def test_tail_work_ledger_preserves_quant_work(self) -> None:
        ledger = overlays.work_ledger(num_tokens=17, topk=8, cols=2048, warps=9)
        self.assertEqual(ledger["logical_routes"], 136)
        self.assertEqual(ledger["productive_claims_baseline"], 8)
        self.assertEqual(ledger["productive_claims_candidate"], 2)
        self.assertEqual(
            ledger["quant_blocks_baseline"], ledger["quant_blocks_candidate"]
        )
        self.assertEqual(
            ledger["expert_scale_lane_loads_baseline"],
            ledger["expert_scale_lane_loads_candidate"],
        )
        self.assertEqual(
            ledger["bf16_block_loads_baseline"],
            8 * ledger["bf16_block_loads_candidate"],
        )


if __name__ == "__main__":
    unittest.main()
