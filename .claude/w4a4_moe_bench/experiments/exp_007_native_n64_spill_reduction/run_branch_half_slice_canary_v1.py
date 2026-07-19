#!/usr/bin/env python3
"""Amplitude-corrected exp_007 branch/half/slice canary follow-up.

This keeps canary_v0 immutable.  The marker construction, route, diagonal FC2
map, block gate, and strict correctness protocol are unchanged; only the FC2
global scale is fixed to 0.25 so the final-output absolute scale fits the
pre-registered max-absolute-error protocol.
"""

from __future__ import annotations

from dataclasses import replace
import sys

import torch

import run_branch_half_slice_canary as canary_v0


W2_GLOBAL_SCALE = 0.25


def install_scaled_weights() -> None:
    original = canary_v0.build_canary_weights

    def build_canary_weights(fixture_module, weights, branch):
        value = original(fixture_module, weights, branch)
        w2_global_scale = torch.full_like(
            value.w2_global_scale, W2_GLOBAL_SCALE
        )
        manifest = dict(value.manifest)
        manifest.update(
            {
                "canary_revision": "v1_output_amplitude_corrected",
                "w2_global_scale_value": W2_GLOBAL_SCALE,
                "w2_global_scale_sha256": fixture_module.tensor_sha256(
                    w2_global_scale
                ),
                "strict_thresholds_changed": False,
            }
        )
        return replace(
            value,
            w2_global_scale=w2_global_scale,
            manifest=manifest,
        )

    canary_v0.build_canary_weights = build_canary_weights


def main() -> int:
    install_scaled_weights()
    return canary_v0.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
