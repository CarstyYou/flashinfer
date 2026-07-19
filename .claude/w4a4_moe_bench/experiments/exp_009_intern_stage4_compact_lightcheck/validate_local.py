#!/usr/bin/env python3
"""CPU-only validation for the exp_009 adapter and source identities."""

from __future__ import annotations

import json

from build_adapter import (
    ADAPTER_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    DIFF_NAME,
    EXPECTED_ORIGINAL_SHA256,
    IDENTITY_NAME,
    sha256_file,
    validate_adapter,
)
from run_exp009_arm import ARM_NAME, EXPECTED_BLOCK


FLASHINFER_ROOT = DEFAULT_SOURCE.parents[2]
PRODUCTION = (
    FLASHINFER_ROOT
    / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
EXPECTED_PRODUCTION_SHA256 = (
    "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106"
)


def main() -> int:
    adapter = DEFAULT_OUTPUT_DIR / ADAPTER_NAME
    diff = DEFAULT_OUTPUT_DIR / DIFF_NAME
    identity_path = DEFAULT_OUTPUT_DIR / IDENTITY_NAME
    for path in (DEFAULT_SOURCE, PRODUCTION, adapter, diff, identity_path):
        if not path.is_file():
            raise RuntimeError(f"missing local contract artifact: {path}")

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    validation = validate_adapter(
        DEFAULT_SOURCE.read_text(encoding="utf-8"),
        adapter.read_text(encoding="utf-8"),
    )
    checks = {
        "original_sha256": sha256_file(DEFAULT_SOURCE) == EXPECTED_ORIGINAL_SHA256,
        "production_sha256": sha256_file(PRODUCTION) == EXPECTED_PRODUCTION_SHA256,
        "adapter_sha256": sha256_file(adapter) == identity["adapter"]["sha256"],
        "diff_sha256": sha256_file(diff) == identity["diff"]["sha256"],
        "identity_original_sha256": identity["original"]["sha256"]
        == EXPECTED_ORIGINAL_SHA256,
        "ast_equal_after_removing_keywords": validation[
            "ast_equal_after_removing_keywords"
        ],
        "only_three_additions": validation["unified_diff_additions"] == 3
        and validation["unified_diff_deletions"] == 0,
        "arm_is_distinct": ARM_NAME not in {"baseline_4warp"},
        "expected_block_is_160_threads": EXPECTED_BLOCK == (160, 1, 1),
    }
    if not all(checks.values()):
        raise RuntimeError(f"local exp_009 validation failed: {checks}")
    print(json.dumps({"status": "pass", "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
