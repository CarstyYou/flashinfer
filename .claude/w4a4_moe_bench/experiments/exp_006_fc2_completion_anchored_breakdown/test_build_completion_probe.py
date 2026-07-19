from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FLASHINFER_ROOT = ROOT.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_completion_probe as builder
from exp006_common import CONTROL, EVENT_ABI, PROBE, TASK_TICKS


def _production() -> tuple[str, str]:
    return (
        (FLASHINFER_ROOT / builder.KERNEL_RELATIVE_PATH).read_text(),
        (FLASHINFER_ROOT / builder.DISPATCH_RELATIVE_PATH).read_text(),
    )


def test_completion_anchors_and_slot_partition() -> None:
    kernel_source, _ = _production()
    kernel = builder._instrument_kernel(kernel_source)
    ast.parse(kernel)

    assert f"_EXP004_TASK_TICKS = {TASK_TICKS}" in kernel
    assert kernel.count(builder._task_store("7", indent=32)) == 1
    assert kernel.count(builder._task_store("8", indent=32)) == 1
    assert kernel.count("Int32(9) + output_tile_idx * Int32(20) + warp_idx") == 1
    assert kernel.count("Int32(13) + output_tile_idx * Int32(20) + warp_idx") == 1
    assert kernel.count("Int32(17) + output_tile_idx * Int32(20) + warp_idx") == 1
    assert kernel.count("Int32(21) + output_tile_idx * Int32(20) + warp_idx") == 1
    assert kernel.count("Int32(25) + output_tile_idx * Int32(20) + warp_idx") == 1
    for event in range(329, 339):
        assert kernel.count(f"+ Int32({event}),") == 1

    c_position = kernel.index("Int32(13) + output_tile_idx * Int32(20) + warp_idx")
    scatter_comment = kernel.index("# Scatter using precomputed metadata", c_position)
    d_position = kernel.index("Int32(17) + output_tile_idx * Int32(20) + warp_idx")
    rows_offset = kernel.index("rows_offset = Int32(epi_m)", d_position)
    e_position = kernel.index("Int32(21) + output_tile_idx * Int32(20) + warp_idx")
    post_barrier_comment = kernel.index("# Post-scatter barrier", e_position)
    f_position = kernel.index("Int32(25) + output_tile_idx * Int32(20) + warp_idx")
    assert c_position < scatter_comment < d_position < rows_offset
    assert rows_offset < e_position < post_barrier_comment < f_position
    f_block = kernel[f_position - 600 : f_position + 300]
    assert "if lane_id == Int32(0):" in f_block
    assert "if warp_idx == Int32(0):" not in f_block


def test_control_and_probe_share_kernel_plumbing(tmp_path: Path) -> None:
    control_dir = tmp_path / CONTROL
    probe_dir = tmp_path / PROBE
    control = builder.build(FLASHINFER_ROOT, control_dir, enabled=False)
    probe = builder.build(FLASHINFER_ROOT, probe_dir, enabled=True)

    assert control["event_abi"] == probe["event_abi"] == EVENT_ABI
    assert control["arm"] == CONTROL
    assert probe["arm"] == PROBE
    assert control["overlay"]["kernel_sha256"] == probe["overlay"]["kernel_sha256"]
    assert control["overlay"]["dispatch_sha256"] != probe["overlay"]["dispatch_sha256"]
    assert (
        "_EXP004_PHASE_PROBE_ENABLED = False"
        in (control_dir / "moe_dispatch.py").read_text()
    )
    assert (
        "_EXP004_PHASE_PROBE_ENABLED = True"
        in (probe_dir / "moe_dispatch.py").read_text()
    )
    assert json.loads((probe_dir / "identity.json").read_text()) == probe


def test_builder_rejects_task_slice_contract_drift() -> None:
    kernel_source, _ = _production()
    drifted = kernel_source.replace("_TASK_SLICE_CHUNK = 1", "_TASK_SLICE_CHUNK = 2")
    try:
        builder._instrument_kernel(drifted)
    except ValueError as error:
        assert "static contract drift" in str(error)
    else:  # pragma: no cover
        raise AssertionError("builder accepted a task-slice ABI drift")
