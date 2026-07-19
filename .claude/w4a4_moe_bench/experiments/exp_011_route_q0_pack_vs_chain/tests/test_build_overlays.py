from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "exp011_build_overlays", ROOT / "build_overlays.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_full_kernel_overlays_are_ast_valid_and_mechanism_locked(tmp_path):
    builder = _load_builder()
    manifest = builder.build(REPO, tmp_path / "overlays")

    assert manifest["launch_contract"] == {
        "grid": [1, 1, 110],
        "block": [160, 1, 1],
    }
    assert set(manifest["arms"]) == {
        f"{variant}/{mode}" for variant in builder.VARIANTS for mode in builder.MODES
    }
    assert all(arm["mechanism_gate"]["gate_pass"] for arm in manifest["arms"].values())

    root = tmp_path / "overlays"
    static = (root / "static_schedule/probe/moe_dynamic_kernel.py").read_text()
    precomputed = (
        root / "precomputed_phys_row/probe/moe_dynamic_kernel.py"
    ).read_text()
    assert "producer_round * Int32(gdim_z)" in static
    assert "get_ptr_as_int64(pair_head, Int32(0))" not in static
    assert "encoded_route >> Int32(8)" in precomputed
    assert precomputed.count("get_ptr_as_int64(expert_write_rows, expert_id)") == 1


def test_probe_and_no_marker_keep_identical_variant_kernel_source(tmp_path):
    builder = _load_builder()
    builder.build(REPO, tmp_path / "overlays")
    for variant in builder.VARIANTS:
        probe = (
            tmp_path / "overlays" / variant / "probe/moe_dynamic_kernel.py"
        ).read_bytes()
        no_marker = (
            tmp_path / "overlays" / variant / "no_marker/moe_dynamic_kernel.py"
        ).read_bytes()
        assert probe == no_marker
