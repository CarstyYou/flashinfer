#!/usr/bin/env python3
"""Build the exp_009 Eric-kernel compatibility overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


EXP_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = EXP_ROOT.parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from breakdown_harness.fragments.eric_stage4_adapter import (  # noqa: E402
    ADAPTER_NAME,
    DIFF_NAME,
    IDENTITY_NAME,
    AdapterError,
    build_adapter as build_adapter_fragment,
    make_adapter,
    sha256_bytes,
    sha256_file,
    validate_adapter,
)

__all__ = (
    "ADAPTER_NAME",
    "AdapterError",
    "DIFF_NAME",
    "IDENTITY_NAME",
    "make_adapter",
    "sha256_bytes",
    "sha256_file",
    "validate_adapter",
)


DEFAULT_SOURCE = BENCH_ROOT / "moe_dyanmice_kernel_ab_stage4_compact.py"
DEFAULT_OUTPUT_DIR = EXP_ROOT / "results/overlays/intern_stage4_compact"
EXPECTED_ORIGINAL_SHA256 = (
    "91034c7cd3b3b9fe8cbde6dbf1bb2c8c13e4261ff9e9e7d642f3ce9d83788768"
)
IDENTITY_SCHEMA = "exp009.intern-stage4-adapter-identity.v1"


def build_adapter(
    *,
    source: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_original_sha256: str = EXPECTED_ORIGINAL_SHA256,
) -> dict[str, Any]:
    return build_adapter_fragment(
        source=source,
        output_dir=output_dir,
        expected_original_sha256=expected_original_sha256,
        identity_schema=IDENTITY_SCHEMA,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    identity = build_adapter(source=args.source, output_dir=args.output_dir)
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
