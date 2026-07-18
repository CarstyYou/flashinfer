from __future__ import annotations

import json

import pytest

from analyze_whole_kernel_timing import (
    ADDITIVE_PHASES,
    SENTINEL,
    TimingContractError,
    aggregate_replays,
    analyze_replay,
    main,
)


def _cta_row(offset: int, *, consumer_exit: int) -> list[int]:
    return [
        100 + offset,
        110 + offset,
        120 + offset,
        130 + offset,
        140 + offset,
        160 + offset,
        170 + offset,
        180 + offset,
        consumer_exit,
        consumer_exit - 10,
        consumer_exit - 20,
        consumer_exit - 30,
        consumer_exit - 100,
        consumer_exit - 50,
    ]


def _task_row(offset: int) -> list[int]:
    row = [SENTINEL] * 65
    row[:7] = [
        200 + offset,
        210 + offset,
        220 + offset,
        260 + offset,
        300 + offset,
        330 + offset,
        800 + offset,
    ]
    for tile in range(16):
        start = 340 + offset + 10 * tile
        row[7 + 3 * tile : 10 + 3 * tile] = [start, start + 5, start + 10]
    w4 = (
        (225, 230),
        (250, 255),
        (305, 310),
        (400, 410),
        (790, 795),
    )
    for interval, (start, end) in enumerate(w4):
        row[55 + 2 * interval : 57 + 2 * interval] = [
            start + offset,
            end + offset,
        ]
    return row


def _payload() -> dict[str, object]:
    return {
        "cta_ticks": [
            _cta_row(0, consumer_exit=1000),
            _cta_row(100, consumer_exit=1100),
        ],
        "task_ticks": [_task_row(0), _task_row(100), [SENTINEL] * 65],
        "task_cta_z": [0, 1, SENTINEL],
        "task_tail": 2,
    }


def test_whole_kernel_denominator_closes_exactly():
    result = analyze_replay(_payload(), replay_id="synthetic")

    assert result["global_wall_ns"] == 1000
    assert result["sm_equivalent_denominator_ns"] == 2000
    assert result["idle_detail"] == {
        "launch_skew_ns": 100,
        "early_finish_idle_ns": 100,
        "combined_ns": 200,
    }
    assert result["phase_totals_ns"]["FC2_gemm"] == 160
    assert result["phase_totals_ns"]["FC2_epilogue_scatter"] == 160
    assert result["phase_totals_ns"]["task_control_final_drain"] == 1040
    assert result["closure"] == {
        "phase_sum_ns": 2000,
        "denominator_ns": 2000,
        "delta_ns": 0,
        "pass": True,
    }
    assert [phase["phase"] for phase in result["phases"]] == list(ADDITIVE_PHASES)
    assert sum(phase["share_pct"] for phase in result["phases"]) == pytest.approx(100.0)


def test_w4_is_non_additive_and_has_overlap_summary():
    result = analyze_replay(_payload())
    w4 = result["w4_non_additive"]

    assert w4["classification"] == "non-additive overlap track"
    assert w4["interval_sums_ns"] == {
        "gate_tma": 10,
        "gate_pass_wait": 10,
        "up_tma": 10,
        "down_tma": 20,
        "final_pass_wait": 10,
    }
    assert w4["interval_sum_ns"] == 60
    assert w4["union_ns"] == 60
    assert w4["overlap_with_additive_ns"] > 0
    assert "w4_non_additive" not in result["phase_totals_ns"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["task_ticks"][0].__setitem__(0, SENTINEL),
            "not exact-fill",
        ),
        (
            lambda payload: payload["task_cta_z"].__setitem__(0, 2),
            "outside grid_z",
        ),
        (
            lambda payload: payload["task_ticks"][0].__setitem__(3, 219),
            "not monotonic",
        ),
        (
            lambda payload: payload["task_ticks"][2].__setitem__(0, 42),
            "not exact sentinel",
        ),
        (
            lambda payload: payload["cta_ticks"][0].__setitem__(4, 129),
            "not monotonic",
        ),
    ],
)
def test_contract_rejects_bad_capture(mutate, match):
    payload = _payload()
    mutate(payload)
    with pytest.raises(TimingContractError, match=match):
        analyze_replay(payload)


def test_aggregate_uses_sum_of_replay_denominators():
    first = analyze_replay(_payload(), replay_id="first")
    second = analyze_replay(_payload(), replay_id="second")
    aggregate = aggregate_replays([first, second])

    assert aggregate["replays"] == 2
    assert aggregate["sm_equivalent_denominator_ns"] == 4000
    assert aggregate["closure"]["pass"]
    assert sum(item["share_pct"] for item in aggregate["phases"]) == pytest.approx(
        100.0
    )


def test_cli_reads_torch_pt_and_writes_json(tmp_path, capsys):
    torch = pytest.importorskip("torch")
    if not all(hasattr(torch, name) for name in ("int32", "int64", "save", "tensor")):
        pytest.skip("a full PyTorch installation is required for .pt CLI coverage")
    payload = _payload()
    capture = tmp_path / "run_0.pt"
    output = tmp_path / "analysis.json"
    torch.save(
        {
            "cta_ticks": torch.tensor(payload["cta_ticks"], dtype=torch.int64),
            "task_ticks": torch.tensor(payload["task_ticks"], dtype=torch.int64),
            "task_cta_z": torch.tensor(payload["task_cta_z"], dtype=torch.int32),
            "task_tail": torch.tensor(payload["task_tail"], dtype=torch.int32),
        },
        capture,
    )

    assert main([str(capture), "--output", str(output)]) == 0
    from_file = json.loads(output.read_text())
    from_stdout = json.loads(capsys.readouterr().out)
    assert from_file == from_stdout
    assert from_file["schema"] == "exp004.whole-kernel-timing.v1"
    assert from_file["aggregate"]["closure"]["pass"]
