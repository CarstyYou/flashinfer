from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
MODULE_PATH = ROOT / "build_arm_overlays.py"
SPEC = importlib.util.spec_from_file_location("exp003_build_arm_overlays", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def production_text() -> str:
    return (REPO / MODULE.TARGET_RELATIVE_PATH).read_text()


def test_builds_all_pre_registered_overlays() -> None:
    source = production_text()
    overlays = MODULE.build_overlays(source)
    assert set(overlays) == {
        "activation_in_place_up",
        "activation_in_place_gate",
        "up_first_attribution",
    }
    for text in overlays.values():
        assert text != source
        ast.parse(text)


@pytest.mark.parametrize(
    ("arm", "destination"),
    [
        ("activation_in_place_up", "up_slice"),
        ("activation_in_place_gate", "gate_slice"),
    ],
)
def test_in_place_arm_changes_only_fc1_gated_conversion(
    arm: str, destination: str
) -> None:
    source = production_text()
    overlay = MODULE.build_overlays(source)[arm]
    assert f"{destination}[elem_idx] = (" in overlay
    assert f"{destination}.load().to(cutlass.BFloat16)" in overlay
    assert "if cutlass.const_expr(not self.is_gated):" in overlay

    fc2_anchor = (
        "                    # PHASE B: Sweep ALL FC2 output tiles using cached sA\n"
    )
    assert source[source.index(fc2_anchor) :] == overlay[overlay.index(fc2_anchor) :]


def test_up_first_swaps_semantics_without_changing_activation_or_fc2() -> None:
    source = production_text()
    overlay = MODULE.build_overlays(source)["up_first_attribution"]
    activation_anchor = "                    # Activation + quant into sA\n"
    producer_anchor = "            elif warp_idx == self.tma_load_warp_id:\n"
    assert (
        source[source.index(activation_anchor) : source.index(producer_anchor)]
        == overlay[overlay.index(activation_anchor) : overlay.index(producer_anchor)]
    )
    assert "up_acc.fill(0.0)" in overlay
    assert "gate_acc.fill(0.0)" in overlay


def test_transform_rejects_source_drift() -> None:
    drifted = production_text().replace(
        MODULE.GATED_ACTIVATION_OLD,
        MODULE.GATED_ACTIVATION_OLD.replace(
            "tRS_rD_slice[elem_idx] = (", "drifted_slice[elem_idx] = ("
        ),
        1,
    )
    with pytest.raises(ValueError, match="gated activation"):
        MODULE.build_overlays(drifted)
