from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture_triton_nsys import (
    EXPECTED_TRITON_CONFIG,
    capture_replays,
    make_topology_preflight,
    stable_runtime_identity,
    validate_graph_topology,
    validate_resolved_config,
    validate_topology_preflight,
)


CANONICAL_NAMES = [
    "void moe_align_block_size_kernel<int>(...)",
    "void count_and_sort_expert_tokens_kernel<int>(...)",
    "vectorized_elementwise_kernel<4, FillFunctor<float>>",
    "per_tensor_absmax_kernel<__nv_bfloat16>",
    "per_tensor_quant_fp8_kernel<bf16, fp8>",
    "fused_moe_kernel",
    "act_and_mul_kernel<bf16>",
    "vectorized_elementwise_kernel<4, FillFunctor<float>>",
    "per_tensor_absmax_kernel<__nv_bfloat16>",
    "per_tensor_quant_fp8_kernel<bf16, fp8>",
    "fused_moe_kernel",
    "moe_sum_reduce_warp_per_token_vec_kernel<8>",
]


def test_graph_topology_assigns_repeated_names_by_ordinal() -> None:
    topology = validate_graph_topology(CANONICAL_NAMES)
    assert topology["node_count"] == 12
    assert topology["nodes"][5]["role"] == "fc1"
    assert topology["nodes"][10]["role"] == "fc2"
    assert topology["nodes"][2]["role"] == "q0_fill"
    assert topology["nodes"][7]["role"] == "q1_fill"
    assert len(topology["fingerprint_sha256"]) == 64


@pytest.mark.parametrize(
    "names",
    [
        CANONICAL_NAMES[:-1],
        [
            *CANONICAL_NAMES[:5],
            CANONICAL_NAMES[6],
            CANONICAL_NAMES[5],
            *CANONICAL_NAMES[7:],
        ],
        [*CANONICAL_NAMES[:5], "CUTLASS grouped GEMM", *CANONICAL_NAMES[6:]],
    ],
)
def test_graph_topology_fails_closed_on_drift(names: list[str]) -> None:
    with pytest.raises(RuntimeError):
        validate_graph_topology(names)


def test_resolved_m8192_config_is_locked() -> None:
    validate_resolved_config(
        {
            "backend": "sglang_legacy_triton_fp8_chain",
            "config": {
                "up": EXPECTED_TRITON_CONFIG.copy(),
                "down": None,
                "down_max_block_m": None,
            },
        }
    )
    drift = EXPECTED_TRITON_CONFIG | {"BLOCK_SIZE_M": 32}
    with pytest.raises(RuntimeError, match="config drift"):
        validate_resolved_config(
            {
                "backend": "sglang_legacy_triton_fp8_chain",
                "config": {"up": drift, "down": None, "down_max_block_m": None},
            }
        )


def topology_preflight_inputs() -> dict:
    return {
        "source_lock": {"capture": {"sha256": "source"}},
        "runtime_identity": {"image_id": "image", "fingerprint_sha256": "runtime"},
        "fixture": {"fixture_sha256": "fixture", "occupancy_sha256": "occupancy"},
        "weights": {"w1_sha256": "w1", "w2_sha256": "w2"},
        "launch_contract": {"backend": "sglang_legacy_triton_fp8_chain"},
        "graph_topology": validate_graph_topology(CANONICAL_NAMES),
        "artifact_fingerprint_sha256": "artifact",
    }


def test_topology_preflight_hash_and_identity_gate() -> None:
    inputs = topology_preflight_inputs()
    preflight = make_topology_preflight(**inputs)
    topology = validate_topology_preflight(
        preflight,
        source_lock=inputs["source_lock"],
        runtime_identity=inputs["runtime_identity"],
        fixture=inputs["fixture"],
        weights=inputs["weights"],
        launch_contract=inputs["launch_contract"],
        artifact_fingerprint_sha256=inputs["artifact_fingerprint_sha256"],
    )
    assert (
        topology["fingerprint_sha256"] == inputs["graph_topology"]["fingerprint_sha256"]
    )

    with pytest.raises(RuntimeError, match="source_lock drift"):
        validate_topology_preflight(
            preflight,
            source_lock={"capture": {"sha256": "changed"}},
            runtime_identity=inputs["runtime_identity"],
            fixture=inputs["fixture"],
            weights=inputs["weights"],
            launch_contract=inputs["launch_contract"],
            artifact_fingerprint_sha256=inputs["artifact_fingerprint_sha256"],
        )


def test_topology_preflight_rejects_tampering() -> None:
    inputs = topology_preflight_inputs()
    preflight = make_topology_preflight(**inputs)
    preflight["graph_topology"]["nodes"][5]["kernel_name"] = "wrong_kernel"
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        validate_topology_preflight(
            preflight,
            source_lock=inputs["source_lock"],
            runtime_identity=inputs["runtime_identity"],
            fixture=inputs["fixture"],
            weights=inputs["weights"],
            launch_contract=inputs["launch_contract"],
            artifact_fingerprint_sha256=inputs["artifact_fingerprint_sha256"],
        )


def test_runtime_preflight_identity_ignores_only_process_local_fields() -> None:
    runtime = {
        "timestamp_unix": 1.0,
        "hostname": "container-a",
        "image_id": "pinned-image",
        "triton_version": "3.4.0",
    }
    first = stable_runtime_identity(runtime, 2377)
    second = stable_runtime_identity(
        {**runtime, "timestamp_unix": 2.0, "hostname": "container-b"}, 2377
    )
    assert first == second
    assert stable_runtime_identity({**runtime, "image_id": "drift"}, 2377) != first


def test_capture_protocol_order_and_correctness_gates() -> None:
    events: list[str] = []

    def record(name: str):
        def callback(*args):
            suffix = f":{args[0]}" if args else ""
            events.append(name + suffix)
            return 0

        return callback

    labels = capture_replays(
        replays=5,
        flush_l2=record("flush"),
        synchronize=record("sync"),
        graph_replay=record("replay"),
        nvtx_push=record("push"),
        nvtx_pop=record("pop"),
        profiler_start=record("start"),
        profiler_stop=record("stop"),
        validate_output_at=record("validate"),
    )

    assert len(labels) == len(set(labels)) == 5
    assert events.count("start") == events.count("stop") == 1
    assert events.count("replay") == 5
    assert [event for event in events if event.startswith("validate:")] == [
        "validate:0",
        "validate:4",
    ]
    for label in labels:
        push_index = events.index(f"push:{label}")
        assert (
            events[push_index - 2 : push_index] == ["flush", "sync"]
            or label == labels[0]
        )
        assert events[push_index + 1 : push_index + 4] == ["replay", "sync", "pop"]
    first_push = events.index(f"push:{labels[0]}")
    assert events[first_push - 3 : first_push] == ["flush", "sync", "start"]
    assert events[-1] == "stop"
    assert events[-2] == "sync"


def test_capture_protocol_stops_profiler_after_replay_failure() -> None:
    events: list[str] = []

    def replay() -> None:
        events.append("replay")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        capture_replays(
            replays=5,
            flush_l2=lambda: events.append("flush"),
            synchronize=lambda: events.append("sync"),
            graph_replay=replay,
            nvtx_push=lambda label: events.append(f"push:{label}"),
            nvtx_pop=lambda: events.append("pop"),
            profiler_start=lambda: events.append("start") or 0,
            profiler_stop=lambda: events.append("stop") or 0,
            validate_output_at=lambda index: events.append(f"validate:{index}"),
        )
    assert "pop" in events
    assert events[-2:] == ["sync", "stop"]
