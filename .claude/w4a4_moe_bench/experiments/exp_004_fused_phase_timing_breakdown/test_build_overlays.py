from __future__ import annotations

import hashlib
from pathlib import Path

from build_overlays import build_overlays
from exp004_common import (
    DISPATCH_RELATIVE_PATH,
    EXPECTED_DISPATCH_SHA256,
    EXPECTED_KERNEL_SHA256,
    KERNEL_RELATIVE_PATH,
    MEASUREMENT_CONTROL,
    NORMAL,
    PROBE,
)


ROOT = Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_overlays_preserves_production_and_locks_delta(tmp_path):
    kernel_before = (ROOT / KERNEL_RELATIVE_PATH).read_bytes()
    dispatch_before = (ROOT / DISPATCH_RELATIVE_PATH).read_bytes()
    manifest = build_overlays(ROOT, tmp_path / "overlays")

    assert hashlib.sha256(kernel_before).hexdigest() == EXPECTED_KERNEL_SHA256
    assert hashlib.sha256(dispatch_before).hexdigest() == EXPECTED_DISPATCH_SHA256
    assert (ROOT / KERNEL_RELATIVE_PATH).read_bytes() == kernel_before
    assert (ROOT / DISPATCH_RELATIVE_PATH).read_bytes() == dispatch_before

    normal = tmp_path / "overlays" / NORMAL
    assert (normal / "moe_dynamic_kernel.py").read_bytes() == kernel_before
    assert (normal / "moe_dispatch.py").read_bytes() == dispatch_before
    assert manifest["arms"][NORMAL]["kernel_byte_identical_to_production"]
    assert manifest["arms"][NORMAL]["dispatch_byte_identical_to_production"]

    control = tmp_path / "overlays" / MEASUREMENT_CONTROL
    probe = tmp_path / "overlays" / PROBE
    assert sha256(control / "moe_dynamic_kernel.py") == sha256(
        probe / "moe_dynamic_kernel.py"
    )
    control_dispatch = (control / "moe_dispatch.py").read_text()
    probe_dispatch = (probe / "moe_dispatch.py").read_text()
    assert (
        control_dispatch.replace(
            "_EXP004_PHASE_PROBE_ENABLED = False",
            "_EXP004_PHASE_PROBE_ENABLED = True",
        )
        == probe_dispatch
    )
    assert "mov.u64 $0, %clock64;" in (probe / "moe_dynamic_kernel.py").read_text()
    assert "phase_probe_enabled=_EXP004_PHASE_PROBE_ENABLED" in probe_dispatch
