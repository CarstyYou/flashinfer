#!/usr/bin/env python3
"""Build the exp_005 8-warp temporal-N64 diagnostic overlay.

The source is derived from the immutable Candidate-A overlay.  It keeps the
8-warp MMA ownership/layout, but holds only one N64 Gate/Up accumulator pair at
a time.  The existing full-N128 TMA descriptors are deliberately replayed once
per half; this first probe trades extra FC1 TMA traffic for a shorter Gate live
range and is therefore a whole design-bundle experiment, not a pure spill
ablation.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANCHOR_NAME = "candidate_8warp_serial_v0"
SUBJECT_NAME = "candidate_8warp_n64_temporal_replay_v0"
ANCHOR_PATH = ROOT / "results" / "overlays" / ANCHOR_NAME / "moe_dynamic_kernel.py"
ANCHOR_SHA256 = "3e2bda4e09dc2c67d97abea4d392eb5fa117de9abdd80c49b0e89e1f2dd0b445"
DEFAULT_OUTPUT = ROOT / "results" / "n64_temporal_replay" / "overlays"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return source.replace(before, after, 1)


def replace_count(source: str, before: str, after: str, count: int, label: str) -> str:
    actual = source.count(before)
    if actual != count:
        raise RuntimeError(f"{label}: expected {count} exact matches, found {actual}")
    return source.replace(before, after)


def indent_block(block: str, spaces: int) -> str:
    prefix = " " * spaces
    return "".join(
        prefix + line if line.strip() else line for line in block.splitlines(True)
    )


def transform(anchor: str) -> tuple[str, list[dict[str, object]]]:
    source = anchor
    records: list[dict[str, object]] = []

    def exact(before: str, after: str, label: str) -> None:
        nonlocal source
        source = replace_once(source, before, after, label)
        records.append({"label": label, "kind": "exact-replacement"})

    exact(
        """        self.is_gated = is_gated_activation(activation)
        self.swiglu_alpha = float(swiglu_alpha)
""",
        """        self.is_gated = is_gated_activation(activation)
        if not self.is_gated:
            raise ValueError(
                "the temporal-N64 experiment overlay is gated-activation only"
            )
        self.swiglu_alpha = float(swiglu_alpha)
""",
        "fail closed outside the registered gated-activation scope",
    )

    exact(
        """        self.pass_final_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
        )
""",
        """        self.pass_final_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
        )
        self.pass_subtile_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_cta,
        )
""",
        "add CTA-wide half-complete barrier",
    )

    exact(
        """        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        gate_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        up_acc = (
            cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            if self.is_gated
            else gate_acc
        )
""",
        """        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        fc1_half_acc_shape = tiled_mma.partition_shape_C(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1] // 2)
        )
        gate_acc = cute.make_rmem_tensor(fc1_half_acc_shape, self.acc_dtype)
        up_acc = (
            cute.make_rmem_tensor(fc1_half_acc_shape, self.acc_dtype)
            if self.is_gated
            else gate_acc
        )
""",
        "materialize a real M128xN64 FC1 accumulator shape",
    )

    exact(
        """                tRS_rD_out = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, cutlass.BFloat16
                )

                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
""",
        """                tRS_rD_out = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, cutlass.BFloat16
                )
                tRS_rAct = cute.make_rmem_tensor(
                    tRS_rD[(None, 0, 0)].shape, self.acc_dtype
                )
                tRS_rAct_out = cute.make_rmem_tensor(
                    tRS_rD_out[(None, 0, 0)].shape, cutlass.BFloat16
                )

                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
                mma_tile_n = (self.tile_shape_mnk[1] // 2) // cute.size(
                    tRS_rGate, mode=[2]
                )
""",
        "add one-microtile activation scratch and preserve MMA tile geometry",
    )

    exact(
        """                fc1_m_tiles = cute.size(tCrA, mode=[1])
                fc1_n_tiles = cute.size(tCrB, mode=[1])
""",
        """                fc1_m_tiles = cute.size(tCrA, mode=[1])
                fc1_half_n_tiles = cute.size(gate_acc, mode=[2])
""",
        "declare full and half FC1 N ownership",
    )

    mainloop_start_marker = "                    # Gate GEMM (inlined to avoid @cute.jit pass-by-value for acc)\n"
    quant_start_marker = "                    # Activation + quant into sA\n"
    start = source.index(mainloop_start_marker)
    quant_start = source.index(quant_start_marker, start)
    mainloop = source[start:quant_start]
    nt_pattern = re.compile(
        r"(?m)^(?P<indent>\s*)for _nt in cutlass\.range_constexpr\(fc1_n_tiles\):$"
    )
    nt_matches = list(nt_pattern.finditer(mainloop))
    if len(nt_matches) != 4:
        raise RuntimeError(
            "limit four Gate/Up GEMM loops to one N64 half: "
            f"expected 4 matches, found {len(nt_matches)}"
        )

    def replace_nt(match: re.Match[str]) -> str:
        indent = match.group("indent")
        inner = indent + "    "
        return (
            indent
            + "for _nt in cutlass.range_constexpr(fc1_half_n_tiles):\n"
            + inner
            + "fc1_n = (\n"
            + inner
            + "    fc1_half * fc1_half_n_tiles + _nt\n"
            + inner
            + ")"
        )

    mainloop = nt_pattern.sub(replace_nt, mainloop)
    mainloop = replace_count(
        mainloop,
        "tCrSFB[None, _nt, k_block_idx]",
        "tCrSFB[None, fc1_n, k_block_idx]",
        4,
        "offset Gate/Up SFB fragment to the selected half",
    )
    mainloop = replace_count(
        mainloop,
        "tCrB[None, _nt, k_block_idx]",
        "tCrB[None, fc1_n, k_block_idx]",
        4,
        "offset Gate/Up B fragment to the selected half",
    )
    activation = """                        # Consume the N64 Gate/Up pair immediately.  The
                        # FP32 Gate lifetime ends before the next half starts.
                        for mma_m in cutlass.range_constexpr(fc1_m_tiles):
                            for mma_n_in_half in cutlass.range_constexpr(
                                fc1_half_n_tiles
                            ):
                                full_mma_n = (
                                    fc1_half * fc1_half_n_tiles + mma_n_in_half
                                )
                                gate_slice = tRS_rGate[
                                    (None, mma_m, mma_n_in_half)
                                ]
                                up_slice = tRS_rUp[(None, mma_m, mma_n_in_half)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rAct)
                                ):
                                    g = alpha_value * gate_slice[elem_idx]
                                    u = alpha_value * up_slice[elem_idx]
                                    tRS_rAct[elem_idx] = gated_activation_f32(
                                        g,
                                        u,
                                        activation=self.activation,
                                        limit=self.swiglu_limit,
                                        alpha=self.swiglu_alpha,
                                        beta=self.swiglu_beta,
                                        fast_math=self.fast_math,
                                    )
                                act_vec = tRS_rAct.load()
                                act_vec = act_vec.to(cutlass.BFloat16)
                                tRS_rAct_out.store(act_vec)
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rAct_out,
                                    tRS_sD[(None, mma_m, full_mma_n, 0)],
                                )

                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if fc1_half == 0:
                            # The DMA warp may now replay FC1 into sA/sSFA/sB
                            # without clobbering the first half's activation.
                            self.pass_subtile_barrier.arrive_unaligned()
"""
    nested_mainloop = (
        "                    # Temporal N64: replay the existing full-tile FC1 TMA\n"
        "                    # pipeline for each output half. Useful OMMA work is unchanged.\n"
        "                    for fc1_half in cutlass.range_constexpr(2):\n"
        + indent_block(mainloop, 4)
        + activation
    )
    source = source[:start] + nested_mainloop + source[quant_start:]
    records.append(
        {
            "label": "temporal Gate->Up->activation loop over two N64 halves",
            "kind": "bounded-structural-rewrite",
        }
    )

    quant_end_marker = "                    # ============================================================\n                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n"
    quant_start = source.index(quant_start_marker)
    quant_end = source.index(quant_end_marker, quant_start)
    quant = """                    # Q1 runs once after both N64 halves are durable in sC.
                    # Running it inside the half loop would let the next FC1 TMA
                    # overwrite sA before FC2 consumes the complete activation.
                    sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
                    packed_cols = Int32(self.tile_shape_mnk[2] // 2)
                    sf_blocks_per_row = Int32(self.tile_shape_mnk[2] // 16)
                    gs_value = global_scale[task_expert_idx].to(cutlass.Float32)
                    if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                        0.0
                    ):
                        if self.fast_math:
                            gs_value = rcp_approx_ftz(gs_value)
                        else:
                            gs_value = cutlass.Float32(1.0) / gs_value

                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        epi_m_valid = valid_rows - Int32(epi_m) * Int32(
                            self.epi_tile[0]
                        )
                        silu_epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                        if epi_m_valid > Int32(0):
                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                            epi_rows = epi_m_valid
                            if epi_rows > Int32(self.epi_tile[0]):
                                epi_rows = Int32(self.epi_tile[0])
                            if epi_rows < Int32(0):
                                epi_rows = Int32(0)
                            quant_idx = Int32(tidx)
                            while quant_idx < epi_rows * sf_blocks_per_row:
                                local_row = quant_idx // sf_blocks_per_row
                                row = rows_offset + local_row
                                sf_block = quant_idx - local_row * sf_blocks_per_row
                                block_start = sf_block * Int32(16)

                                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = cutlass.Float32(
                                        sC[
                                            local_row,
                                            block_start + elem_idx,
                                            silu_epi_buffer,
                                        ]
                                    )
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))

                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )
                                packed_base = sf_block << Int32(3)
                                dst_pcol = row & Int32(63)
                                xor_bits = (
                                    ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
                                )
                                row_high = row >> Int32(6)
                                for byte_idx in cutlass.range_constexpr(8):
                                    src_pcol = packed_base + Int32(byte_idx)
                                    dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
                                    dst_flat = dst_row * packed_cols + dst_pcol
                                    byte_val = Uint8(
                                        (packed64 >> Uint64(byte_idx * 8)) & Uint64(0xFF)
                                    )
                                    sA_u8[dst_flat] = byte_val

                                outer_m_idx = row % Int32(32)
                                inner_m_idx = row // Int32(32)
                                inner_k_idx = sf_block % Int32(4)
                                k_tile_idx = sf_block // Int32(4)
                                sf_raw_idx = (
                                    k_tile_idx * Int32(32 * 4 * 4)
                                    + outer_m_idx * Int32(4 * 4)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                st_shared_u8(sfa_base_addr + sf_raw_idx, scale_byte)
                                quant_idx += Int32(
                                    self.num_mma_warps * self.num_threads_per_warp
                                )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

"""
    source = source[:quant_start] + quant + source[quant_end:]
    records.append(
        {
            "label": "move Q1 after both N64 halves",
            "kind": "bounded-structural-rewrite",
        }
    )

    producer_start_marker = "                    # ---- FC1 gate/only pass ----\n"
    producer_end_marker = (
        "                    # ---- FC2 B_down loads: continuous pipeline ----\n"
    )
    producer_start = source.index(producer_start_marker)
    producer_end = source.index(producer_end_marker, producer_start)
    producer = source[producer_start:producer_end]
    nested_producer = (
        "                    # Replay the existing full-N128 FC1 TMA descriptors once\n"
        "                    # per temporal N64 output half.\n"
        "                    for fc1_half in cutlass.range_constexpr(2):\n"
        + indent_block(producer, 4)
        + "                        if fc1_half == 0:\n"
        + "                            self.pass_subtile_barrier.wait_unaligned()\n\n"
    )
    source = source[:producer_start] + nested_producer + source[producer_end:]
    records.append(
        {
            "label": "replay full FC1 TMA producers for two temporal halves",
            "kind": "bounded-structural-rewrite",
        }
    )

    if "fc1_n_tiles" in source:
        raise RuntimeError("stale fc1_n_tiles reference remains after transformation")
    return source, records


def build(anchor_path: Path, output_dir: Path) -> dict[str, object]:
    anchor_bytes = anchor_path.read_bytes()
    actual_anchor_hash = sha256(anchor_bytes)
    if actual_anchor_hash != ANCHOR_SHA256:
        raise RuntimeError(
            f"Candidate-A hash drift: expected {ANCHOR_SHA256}, got {actual_anchor_hash}"
        )
    anchor = anchor_bytes.decode("utf-8")
    subject, transforms = transform(anchor)
    ast.parse(subject, filename=f"{SUBJECT_NAME}/moe_dynamic_kernel.py")
    subject_bytes = subject.encode("utf-8")

    anchor_out = output_dir / ANCHOR_NAME / "moe_dynamic_kernel.py"
    subject_out = output_dir / SUBJECT_NAME / "moe_dynamic_kernel.py"
    diff_out = output_dir / f"{SUBJECT_NAME}.diff"
    identity_out = output_dir / "identity.json"
    anchor_out.parent.mkdir(parents=True, exist_ok=True)
    subject_out.parent.mkdir(parents=True, exist_ok=True)
    anchor_out.write_bytes(anchor_bytes)
    subject_out.write_bytes(subject_bytes)
    diff = "".join(
        difflib.unified_diff(
            anchor.splitlines(keepends=True),
            subject.splitlines(keepends=True),
            fromfile=f"{ANCHOR_NAME}/moe_dynamic_kernel.py",
            tofile=f"{SUBJECT_NAME}/moe_dynamic_kernel.py",
        )
    ).encode("utf-8")
    diff_out.write_bytes(diff)
    identity = {
        "schema": "exp005.temporal-n64-overlay.v1",
        "relationship": "R2_temporal_n64_replay_vs_candidateA",
        "anchor": {
            "name": ANCHOR_NAME,
            "source": str(anchor_path.resolve()),
            "path": str(anchor_out.resolve()),
            "sha256": actual_anchor_hash,
        },
        "subject": {
            "name": SUBJECT_NAME,
            "path": str(subject_out.resolve()),
            "sha256": sha256(subject_bytes),
            "ast_parse": "pass",
            "transforms": transforms,
        },
        "diff": {
            "path": str(diff_out.resolve()),
            "sha256": sha256(diff),
            "size_bytes": len(diff),
        },
    }
    identity_out.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, default=ANCHOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.anchor, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
