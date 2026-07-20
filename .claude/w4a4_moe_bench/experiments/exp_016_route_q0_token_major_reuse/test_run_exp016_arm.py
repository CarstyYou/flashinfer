from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_exp016_arm as target  # noqa: E402


def test_locked_scale_patterns_remain_expert_shaped() -> None:
    equal = target.make_input_scale(8, "equal")
    unequal = target.make_input_scale(8, "unequal")

    assert equal.shape == (8,)
    assert equal.tolist() == [1.0] * 8
    assert unequal.shape == (8,)
    assert unequal.tolist() == [0.5, 1.0, 2.0, 1.0] * 2


def test_unequal_input_scale_does_not_change_oracle_weight_representation() -> None:
    @dataclass(frozen=True)
    class Weights:
        w1_global_scale: torch.Tensor
        manifest: dict[str, object]

    runtime = Weights(
        torch.tensor([0.5, 1.0, 2.0, 1.0], dtype=torch.float32),
        {},
    )
    oracle = target.reference_weights_for_input_scale(runtime, "unequal")

    assert runtime.w1_global_scale.tolist() == [0.5, 1.0, 2.0, 1.0]
    assert oracle.w1_global_scale.tolist() == [1.0] * 4
    assert "exp016_oracle_w1_scale_semantics" in oracle.manifest


@pytest.mark.parametrize(
    ("arm", "m", "unit", "claim", "productive", "terminal"),
    (
        (target.BASELINE, 256, "routed_pair", 18, 114, 4032),
        (target.BASELINE, 8192, "routed_pair", 18, 3641, 67518),
        (target.CANDIDATE, 256, "token", 9, 29, 1251),
        (target.CANDIDATE, 8192, "token", 9, 911, 9189),
    ),
)
def test_producer_counter_contract(
    arm: str,
    m: int,
    unit: str,
    claim: int,
    productive: int,
    terminal: int,
) -> None:
    contract = target.producer_counter_contract(arm, m)
    assert contract["unit"] == unit
    assert contract["claim"] == claim
    assert contract["productive_claims"] == productive
    assert contract["terminal"] == terminal


def test_scale_storage_offsets_cover_two_complete_tiles() -> None:
    rows = torch.arange(256, dtype=torch.int64)
    offsets = target.scale_storage_offsets(rows, 128)

    assert offsets.shape == (256, 128)
    flattened = offsets.flatten()
    assert torch.unique(flattened).numel() == flattened.numel()
    assert int(flattened.min()) == 0
    assert int(flattened.max()) == flattened.numel() - 1


def _fixture_and_workspace(reverse_e1: bool) -> tuple[SimpleNamespace, SimpleNamespace]:
    topk_ids = torch.tensor([[1, 0], [2, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)
    fixture = SimpleNamespace(
        m=2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    row_counts = torch.tensor([1, 2, 1], dtype=torch.int32)
    expert_tile_base = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    token_map = torch.full((384,), -1, dtype=torch.int32)
    token_weights = torch.zeros(384, dtype=torch.float32)
    packed_input = torch.zeros((1, 384, 8), dtype=torch.uint8)
    packed_scale = torch.zeros((384, 4), dtype=torch.uint8)

    routes = {
        (0, 0): (0.75, 10, 20),
        (1, 0): (0.25, 11, 21),
        (1, 1): (0.6, 12, 22),
        (2, 1): (0.4, 13, 23),
    }
    e1_tokens = (1, 0) if reverse_e1 else (0, 1)
    placements = {
        (0, 0): 0,
        (1, e1_tokens[0]): 128,
        (1, e1_tokens[1]): 129,
        (2, 1): 256,
    }
    for key, physical_row in placements.items():
        weight, packed_code, scale_code = routes[key]
        token_map[physical_row] = key[1]
        token_weights[physical_row] = weight
        packed_input[0, physical_row].fill_(packed_code)
        scale_offset = target.scale_storage_offsets(
            torch.tensor([physical_row]), 1
        ).item()
        packed_scale.view(-1)[scale_offset] = scale_code

    workspace = SimpleNamespace(
        row_counts=row_counts,
        expert_tile_base=expert_tile_base,
        token_map=token_map,
        token_weights=token_weights,
        packed_input=packed_input,
        packed_input_scale=packed_scale,
    )
    return fixture, workspace


def test_logical_payload_digest_is_physical_row_order_independent() -> None:
    fixture_a, workspace_a = _fixture_and_workspace(reverse_e1=True)
    fixture_b, workspace_b = _fixture_and_workspace(reverse_e1=False)

    digest_a = target.canonical_logical_payload_digest(
        workspace_a, fixture_a, hidden_size=16
    )
    digest_b = target.canonical_logical_payload_digest(
        workspace_b, fixture_b, hidden_size=16
    )

    assert digest_a["gate_pass"]
    assert digest_b["gate_pass"]
    assert digest_a["logical_routes"] == 4
    assert digest_a["combined_sha256"] == digest_b["combined_sha256"]
    assert digest_a["packed_fp4_sha256"] == digest_b["packed_fp4_sha256"]
    assert digest_a["sfa_sha256"] == digest_b["sfa_sha256"]


def test_route_plan_rejects_duplicate_expert_per_token() -> None:
    fixture, workspace = _fixture_and_workspace(reverse_e1=False)
    fixture.topk_ids[0] = torch.tensor([1, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="duplicate expert"):
        target.canonical_route_plan(
            row_counts=workspace.row_counts,
            expert_tile_base=workspace.expert_tile_base,
            token_map=workspace.token_map,
            token_weights=workspace.token_weights,
            topk_ids=fixture.topk_ids,
            topk_weights=fixture.topk_weights,
        )
