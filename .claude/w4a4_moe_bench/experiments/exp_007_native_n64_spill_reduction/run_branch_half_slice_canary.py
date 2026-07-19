#!/usr/bin/env python3
"""Run an exp_007 branch/half/slice address canary through the exp_005 worker.

The ordinary deterministic-random fixture is a broad correctness check, but it
does not make a particular FC1 N64 address mistake easy to diagnose.  This
wrapper keeps the independently implemented PyTorch oracle and launch harness,
while replacing the payload with two directed variants:

* ``up``:   eight Up N64 ranges carry distinct FP4 codes; Gate is constant.
* ``gate``: eight Gate N64 ranges carry distinct FP4 codes; Up is constant.

FC2 is an FP4 diagonal map for the first 512 output coordinates.  Consequently
each logical FC1 N64 range appears in a separate 64-wide output block.  Every
block must be nonzero and must independently agree with the oracle.  This is an
address/coverage diagnostic; it is not a performance workload.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parent
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
sys.path.insert(0, str(EXP005))

import run_exp005_arm as worker  # noqa: E402


N64 = 64
INTERMEDIATE = 512
# E2M1 codes: +0.5,+1,+1.5,+2,+3,+4,+6,-0.5.  Keeping every marker
# nonzero lets the diagnostic distinguish a missed write from a valid marker.
MARKER_CODES = (1, 2, 3, 4, 5, 6, 7, 9)
FIXED_CODE = 2
BLOCK_RELATIVE_L2_LIMIT = 0.15


def packed_byte(code: int) -> int:
    if not 0 <= code < 16:
        raise ValueError(code)
    return code | (code << 4)


def build_canary_weights(fixture_module: Any, weights: Any, branch: str):
    w1 = torch.empty_like(weights.w1_packed)
    w1.fill_(packed_byte(FIXED_CODE))

    branch_base = 0 if branch == "up" else INTERMEDIATE
    for segment, code in enumerate(MARKER_CODES):
        begin = branch_base + segment * N64
        end = begin + N64
        w1[:, begin:end, :].fill_(packed_byte(code))

    # Row k of FC2 reads only activation channel k.  The remaining H-I output
    # rows stay zero, so each N64 marker is independently visible in output.
    w2 = torch.zeros_like(weights.w2_packed)
    one_code = FIXED_CODE
    for k in range(INTERMEDIATE):
        byte_index = k // 2
        nibble = one_code if k % 2 == 0 else one_code << 4
        w2[:, k, byte_index] = nibble

    manifest = dict(weights.manifest)
    manifest.update(
        {
            "fixture_kind": f"branch_half_slice_canary_{branch}",
            "marker_branch": branch,
            "marker_n64_codes": list(MARKER_CODES),
            "fixed_other_branch_code": FIXED_CODE,
            "fc2_map": "FP4 diagonal: activation k -> output k for k in [0,512)",
            "w1_packed_sha256": fixture_module.tensor_sha256(w1),
            "w2_packed_sha256": fixture_module.tensor_sha256(w2),
        }
    )
    return replace(weights, w1_packed=w1, w2_packed=w2, manifest=manifest)


def install_canary(branch: str) -> None:
    original_make_case = worker.make_case

    def make_case(args):
        fixture_module, fixture, weights = original_make_case(args)
        # Positive constant input avoids cancellation and makes all eight
        # marker blocks observable after Q0/FC1/Q1/FC2.
        x = torch.full_like(fixture.x, 0.125)
        fixture_manifest = dict(fixture.manifest)
        fixture_manifest.update(
            {
                "fixture_kind": f"branch_half_slice_canary_{branch}",
                "x_pattern": "constant_bf16_0.125",
                "x_sha256": fixture_module.tensor_sha256(x),
            }
        )
        fixture = fixture_module.RoutedFixture(
            fixture.m,
            x,
            fixture.topk_ids,
            fixture.topk_weights,
            fixture_manifest,
        )
        weights = build_canary_weights(fixture_module, weights, branch)

        original_diagnostics = fixture_module.output_diagnostics

        def canary_diagnostics(actual: torch.Tensor, reference: torch.Tensor):
            result = original_diagnostics(actual, reference)
            block_rows = []
            all_blocks_pass = True
            for segment in range(len(MARKER_CODES)):
                begin = segment * N64
                end = begin + N64
                actual_block = actual[:, begin:end].float()
                reference_block = reference[:, begin:end].float()
                reference_norm = torch.linalg.vector_norm(reference_block)
                relative_l2 = torch.linalg.vector_norm(
                    actual_block - reference_block
                ) / reference_norm.clamp_min(1.0e-12)
                nonzero = bool(reference_norm.item() > 1.0e-6)
                block_pass = nonzero and bool(
                    relative_l2.item() <= BLOCK_RELATIVE_L2_LIMIT
                )
                all_blocks_pass = all_blocks_pass and block_pass
                block_rows.append(
                    {
                        "segment": segment,
                        "global_n_range": [begin, end],
                        "marker_code": MARKER_CODES[segment],
                        "reference_l2": float(reference_norm.item()),
                        "relative_l2": float(relative_l2.item()),
                        "pass": block_pass,
                    }
                )
            result["canary_branch"] = branch
            result["canary_block_relative_l2_limit"] = BLOCK_RELATIVE_L2_LIMIT
            result["canary_blocks"] = block_rows
            result["canary_all_blocks_pass"] = all_blocks_pass
            result["formal_pass"] = bool(result["formal_pass"] and all_blocks_pass)
            return result

        fixture_module.output_diagnostics = canary_diagnostics
        return fixture_module, fixture, weights

    worker.make_case = make_case


def parse_wrapper_args(argv: Sequence[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--canary", choices=("up", "gate"), required=True)
    parsed, remaining = parser.parse_known_args(argv)
    return parsed.canary, remaining


def main(argv: Sequence[str] | None = None) -> int:
    branch, worker_argv = parse_wrapper_args(list(argv or sys.argv[1:]))
    install_canary(branch)
    return worker.main(worker_argv)


if __name__ == "__main__":
    raise SystemExit(main())
