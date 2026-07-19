#!/usr/bin/env python3
"""Final-output-scaled exp_007 branch/half/slice canary follow-up.

The kernel inputs, W1 markers, Q1, and diagonal FC2 payload are identical to
canary_v0.  Only token_final_scales are multiplied by 0.25, so the already
computed expert outputs are linearly scaled at the final scatter boundary.
This keeps the address diagnostic intact while fitting the pre-registered
absolute-error protocol.
"""

from __future__ import annotations

import sys

import run_branch_half_slice_canary as canary_v0


SCATTER_OUTPUT_SCALE = 0.25


def install_scaled_route_weights() -> None:
    original_install = canary_v0.install_canary

    def install_canary(branch: str) -> None:
        original_install(branch)
        original_make_case = canary_v0.worker.make_case

        def make_case(args):
            fixture_module, fixture, weights = original_make_case(args)
            topk_weights = (
                fixture.topk_weights * SCATTER_OUTPUT_SCALE
            ).contiguous()
            manifest = dict(fixture.manifest)
            manifest.update(
                {
                    "canary_revision": "v2_final_scatter_scale",
                    "scatter_output_scale": SCATTER_OUTPUT_SCALE,
                    "topk_weights_sha256": fixture_module.tensor_sha256(
                        topk_weights
                    ),
                    "topk_weight_sum": float(topk_weights[0].sum().item()),
                    "strict_thresholds_changed": False,
                    "kernel_compute_payload_changed_from_v0": False,
                }
            )
            fixture = fixture_module.RoutedFixture(
                fixture.m,
                fixture.x,
                fixture.topk_ids,
                topk_weights,
                manifest,
            )
            return fixture_module, fixture, weights

        canary_v0.worker.make_case = make_case

    canary_v0.install_canary = install_canary


def main() -> int:
    install_scaled_route_weights()
    return canary_v0.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
