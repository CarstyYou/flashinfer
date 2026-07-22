"""Compatibility import for the canonical W4A4 MoE case harness."""

from pathlib import Path
import sys


BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from breakdown_harness.case import *  # noqa: E402,F403
