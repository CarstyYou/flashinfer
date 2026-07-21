#!/usr/bin/env python3
"""Unit tests for check_phase_structure.py (no CuTeDSL/GPU required)."""

import sys
import textwrap
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_phase_structure as gate  # noqa: E402


VALID_SOURCE = textwrap.dedent(
    """
    class MoEDynamicKernel:
        def __init__(self, mma_tiler_mn, tile_k):
            self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
            self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])

        @cute.jit
        def resident_grid_barrier(self):
            pass

        @cute.jit
        def publish_ready_tasks(self):
            pass

        @cute.jit
        def publish_deferred_tasks(self):
            pass

        @cute.jit
        def claim_and_cache_task(self):
            pass

        @cute.jit
        def fc1_gate_up_swiglu_to_sC(self, cons_state, ml_pipeline, fc1_half):
            gate_acc = cute.make_rmem_tensor(shape, dtype)
            up_acc = cute.make_rmem_tensor(shape, dtype)
            cute.gemm(mma, gate_acc, lhs, rhs, gate_acc)
            cute.gemm(mma, up_acc, lhs, rhs, up_acc)
            cons_state.advance()
            ml_pipeline.consumer_release(cons_state)
            if fc1_half == 1:
                self.pass_gate_barrier.arrive_unaligned()
            gated_activation_f32(gate_acc, up_acc)
            return cons_state

        @cute.jit
        def quantize_q1_sC_to_sA_sSFA(self, sC):
            quantize(sC)

        @cute.jit
        def load_fc2_a_fragments(self, sC):
            preload(sC)

        @cute.jit
        def fc2_to_sC(self, phase2_cons_state, sC):
            down_acc = cute.make_rmem_tensor(shape, dtype)
            cute.gemm(mma, down_acc, lhs, rhs, down_acc)
            epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
            for epi_m in cutlass.range_constexpr(epi_rest_m):
                cute.copy(down_acc, sC)
            phase2_cons_state.advance()
            return phase2_cons_state

        @cute.jit
        def scatter_sC_to_gmem(self, sC, scatter_output):
            epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
            for epi_m in cutlass.range_constexpr(epi_rest_m):
                value = sC[0]
                scatter_output[0] = value

        @cute.jit
        def load_fc1_tma_slice(self, prod_state):
            prod_state.advance()
            return prod_state

        @cute.jit
        def load_fc2_tma_slice(self, phase2_prod_state):
            phase2_prod_state.advance()
            return phase2_prod_state

        @cute.jit
        def initialize_route_q0_and_publish(self):
            pass

        @cute.jit
        def __call__(self):
            self.kernel()

        @cute.kernel
        def kernel(self):
            while slice_idx < slice_count:
                cons_state = self.fc1_gate_up_swiglu_to_sC(
                    cons_state, ml_pipeline, fc1_half
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()
                self.quantize_q1_sC_to_sA_sSFA(sC)
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()
                self.load_fc2_a_fragments(sC)
                phase2_cons_state.reset_count()
                for output_tile_idx in range(output_tile_count):
                    phase2_cons_state = self.fc2_to_sC(phase2_cons_state, sC)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    self.scatter_sC_to_gmem(sC, scatter_output)
                    self.epilog_sync_barrier.arrive_and_wait()
                self.pass_final_barrier.arrive_unaligned()
                prod_state = self.load_fc1_tma_slice(prod_state)
                phase2_prod_state = self.load_fc2_tma_slice(phase2_prod_state)
    """
)


def remove_nth(text, needle, occurrence):
    start = 0
    for _unused_index in range(occurrence):
        found = text.find(needle, start)
        if found < 0:
            raise AssertionError("fixture needle not found")
        start = found + len(needle)
    return text[:found] + text[found + len(needle) :]


class PhaseStructureGateTests(unittest.TestCase):
    def assert_rejected_with(self, source, expected):
        errors = gate.validate_source(source, "fixture.py")
        self.assertTrue(errors, "mutated fixture unexpectedly passed")
        self.assertTrue(
            any(expected in error for error in errors),
            "expected {!r} in errors: {}".format(expected, errors),
        )

    def test_valid_structure_passes(self):
        self.assertEqual(gate.validate_source(VALID_SOURCE, "fixture.py"), [])

    def test_parse_error_fails_closed(self):
        self.assert_rejected_with("class MoEDynamicKernel(", "AST parse failed")

    def test_missing_helper_is_rejected(self):
        source = VALID_SOURCE.replace(
            "def scatter_sC_to_gmem", "def missing_scatter_sC_to_gmem", 1
        )
        self.assert_rejected_with(source, "missing required helper/method")

    def test_pipeline_state_must_be_rebound(self):
        source = VALID_SOURCE.replace(
            "phase2_cons_state = self.fc2_to_sC(phase2_cons_state, sC)",
            "self.fc2_to_sC(phase2_cons_state, sC)",
            1,
        )
        self.assert_rejected_with(source, "must explicitly rebind")

    def test_accumulator_cannot_escape_its_owner(self):
        source = VALID_SOURCE.replace(
            "value = sC[0]", "gate_acc = 0\n            value = sC[0]", 1
        )
        self.assert_rejected_with(source, "gate_acc must be local only")

    def test_fc2_scatter_order_is_rejected_when_reversed(self):
        old = (
            "                phase2_cons_state = self.fc2_to_sC(phase2_cons_state, sC)\n"
            '                cute.arch.fence_proxy("async.shared", space="cta")\n'
            "                self.epilog_sync_barrier.arrive_and_wait()\n"
            "                self.scatter_sC_to_gmem(sC, scatter_output)\n"
        )
        new = (
            "                self.scatter_sC_to_gmem(sC, scatter_output)\n"
            '                cute.arch.fence_proxy("async.shared", space="cta")\n'
            "                self.epilog_sync_barrier.arrive_and_wait()\n"
            "                phase2_cons_state = self.fc2_to_sC(phase2_cons_state, sC)\n"
        )
        self.assertIn(old, VALID_SOURCE)
        self.assert_rejected_with(
            VALID_SOURCE.replace(old, new, 1), "caller phase order"
        )

    def test_each_phase_handoff_requires_fence_and_barrier(self):
        fence = '                cute.arch.fence_proxy("async.shared", space="cta")\n'
        source = remove_nth(VALID_SOURCE, fence, 1)
        self.assert_rejected_with(source, "fc2 -> scatter must contain exactly one")

    def test_output_tile_requires_post_scatter_barrier(self):
        barrier = "                self.epilog_sync_barrier.arrive_and_wait()\n"
        source = remove_nth(VALID_SOURCE, barrier, 2)
        self.assert_rejected_with(source, "post-scatter epilog")

    def test_fc2_state_reset_is_once_per_slice(self):
        source = VALID_SOURCE.replace(
            "            phase2_cons_state.reset_count()\n", "", 1
        )
        self.assert_rejected_with(source, "reset_count must run exactly once")

    def test_pass_gate_must_precede_swiglu(self):
        old = (
            "        if fc1_half == 1:\n"
            "            self.pass_gate_barrier.arrive_unaligned()\n"
            "        gated_activation_f32(gate_acc, up_acc)\n"
        )
        new = (
            "        gated_activation_f32(gate_acc, up_acc)\n"
            "        if fc1_half == 1:\n"
            "            self.pass_gate_barrier.arrive_unaligned()\n"
        )
        self.assertIn(old, VALID_SOURCE)
        self.assert_rejected_with(
            VALID_SOURCE.replace(old, new, 1),
            "after the final FC1 release and before SwiGLU",
        )

    def test_generic_ctx_is_rejected(self):
        source = VALID_SOURCE.replace(
            "value = sC[0]", "ctx = sC\n            value = sC[0]", 1
        )
        self.assert_rejected_with(source, "generic identifier 'ctx' is forbidden")

    def test_leading_underscore_jit_helper_is_rejected(self):
        insertion = "    @cute.jit\n    def _hidden_phase(self):\n        pass\n\n"
        source = VALID_SOURCE.replace(
            "    @cute.kernel\n    def kernel(self):",
            insertion + "    @cute.kernel\n    def kernel(self):",
            1,
        )
        self.assert_rejected_with(source, "must not have a leading underscore")

    def test_scatter_must_visibly_read_sC(self):
        source = VALID_SOURCE.replace("value = sC[0]", "value = 0", 1)
        self.assert_rejected_with(source, "must visibly read from sC")

    def test_epilogue_m_must_share_tile_m_origin(self):
        source = VALID_SOURCE.replace(
            "self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])",
            "self.epi_tile = (64, mma_tiler_mn[1])",
            1,
        )
        self.assert_rejected_with(source, "must both originate from mma_tiler_mn[0]")

    def test_fc2_and_scatter_must_lock_epi_rest_ratio(self):
        source = VALID_SOURCE.replace(
            "epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]",
            "epi_rest_m = 2",
            1,
        )
        self.assert_rejected_with(source, "fc2_to_sC must assign exactly")


if __name__ == "__main__":
    unittest.main()
