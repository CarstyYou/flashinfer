from types import SimpleNamespace

import pytest

from run_exp009_arm import (
    ARM_NAME,
    EXPECTED_BLOCK,
    EXPECTED_PYTHON_DEPS_SHA256,
    FULL_M_VALUES,
    WorkerAdapterError,
    _failure_classification,
    register_arm,
)


def _fake_modules():
    arms = (
        "baseline_4warp",
        "candidate_8warp_serial_v0",
        "candidate_8warp_n64_temporal_replay_v0",
    )
    common = SimpleNamespace(
        KNOWN_ARMS=arms,
        EXPECTED_BLOCKS={
            "baseline_4warp": (160, 1, 1),
            "candidate_8warp_serial_v0": (288, 1, 1),
            "candidate_8warp_n64_temporal_replay_v0": (288, 1, 1),
        },
    )

    def expected_block(arm):
        return common.EXPECTED_BLOCKS[arm]

    def require_arm_m(arm, m):
        if arm not in common.KNOWN_ARMS or m not in common.M_VALUES:
            raise ValueError((arm, m))

    worker = SimpleNamespace(
        KNOWN_ARMS=arms,
        M_VALUES=(256, 1024, 8192),
        CANDIDATE="candidate_8warp_serial_v0",
        expected_block=expected_block,
        require_arm_m=require_arm_m,
    )
    common.M_VALUES = (256, 1024, 8192)
    return common, worker


def test_registers_distinct_160_thread_candidate_idempotently():
    common, worker = _fake_modules()
    register_arm(common, worker)
    assert common.KNOWN_ARMS[-1] == ARM_NAME
    assert worker.KNOWN_ARMS == common.KNOWN_ARMS
    assert common.EXPECTED_BLOCKS[ARM_NAME] == EXPECTED_BLOCK
    assert worker.CANDIDATE == ARM_NAME
    assert common.EXPECTED_PYTHON_DEPS_SHA256 == EXPECTED_PYTHON_DEPS_SHA256
    assert worker.EXPECTED_PYTHON_DEPS_SHA256 == EXPECTED_PYTHON_DEPS_SHA256
    assert common.M_VALUES == FULL_M_VALUES
    assert worker.M_VALUES == FULL_M_VALUES
    for m in FULL_M_VALUES:
        worker.require_arm_m(ARM_NAME, m)

    register_arm(common, worker)
    assert common.KNOWN_ARMS.count(ARM_NAME) == 1


def test_refuses_worker_common_arm_drift():
    common, worker = _fake_modules()
    worker.KNOWN_ARMS = worker.KNOWN_ARMS[:-1]
    with pytest.raises(WorkerAdapterError, match="KNOWN_ARMS identity drift"):
        register_arm(common, worker)


def test_refuses_conflicting_existing_block():
    common, worker = _fake_modules()
    common.KNOWN_ARMS += (ARM_NAME,)
    worker.KNOWN_ARMS = common.KNOWN_ARMS
    common.EXPECTED_BLOCKS[ARM_NAME] = (288, 1, 1)
    with pytest.raises(WorkerAdapterError, match="block drift"):
        register_arm(common, worker)


def test_refuses_worker_common_m_value_drift():
    common, worker = _fake_modules()
    worker.M_VALUES = (256, 8192)
    with pytest.raises(WorkerAdapterError, match="M_VALUES identity drift"):
        register_arm(common, worker)


@pytest.mark.parametrize(
    ("diagnostics", "workspace", "expected"),
    [
        (
            [{"formal_pass": False, "sentinel_nan_remaining": 0}],
            [],
            "numerical_accuracy_failure",
        ),
        (
            [{"formal_pass": False, "sentinel_nan_remaining": 8}],
            [],
            "sentinel_or_incomplete_output",
        ),
        (
            [{"formal_pass": True}],
            [{"verification": {"gate_pass": False}}],
            "workspace_route_task_failure",
        ),
        ([], [], "runtime_or_harness_failure"),
    ],
)
def test_failure_classification(diagnostics, workspace, expected):
    assert _failure_classification(diagnostics, workspace) == expected
