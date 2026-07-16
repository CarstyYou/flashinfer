from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_exp003.py"
SPEC = importlib.util.spec_from_file_location("exp003_analyzer", MODULE_PATH)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


KERNEL = "generated_MoEDynamicKernel_candidate"
PHASE_IDS = {"gate": 1, "up": 2, "fc2": 3}


def _phase_payload(phase: str, ordinal: int) -> int:
    return PHASE_IDS[phase] * 1_000_000 + ordinal + 1


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _location(warp: int, cta_z: int = 0) -> dict:
    return {
        "gpcId": 0,
        "tpcId": 0,
        "smId": 0,
        "ctaId": [0, 0, cta_z],
        "warpId": warp,
    }


def _range(
    name_index: int,
    location_index: int,
    start: int,
    end: int,
    *,
    payload: int | None = None,
) -> dict:
    start_event = {"eventId": name_index + 1, "timestamp": start}
    if payload is not None:
        start_event.update(payloadType=5, payloadVal=payload)
    return {
        "rangeNameIdx": name_index,
        "rangeScope": 0,
        "rangeType": 2,
        "startTs": start,
        "endTs": end,
        "internalEvents": [start_event, {"eventId": 999, "timestamp": end}],
        "warpLocIdxs": [location_index, location_index],
    }


def _task_ranges(names: list[str], warp: int, *, long_wait: bool = False) -> list[dict]:
    index = {name: names.index(name) for name in names}
    loc = warp
    ranges = [
        _range(index["moe_kernel"], loc, 0, 1000),
        _range(index["marker_calibration"], loc, 1, 2, payload=warp),
        _range(index["phase0_init"], loc, 3, 4),
        _range(index["histogram"], loc, 4, 5),
        _range(index["prefix_sum"], loc, 5, 6),
        _range(index["route_pack"], loc, 6, 7),
        _range(index["setup_compute"], loc, 7, 8),
    ]
    if warp == 4:
        return ranges
    gate_end = 700 if long_wait else 100
    up_start, up_end = (710, 780) if long_wait else (110, 180)
    act_start, act_end = (781, 800) if long_wait else (181, 200)
    fc2_start, fc2_end = (810, 845) if long_wait else (210, 245)
    ranges.extend(
        [
            _range(index["task_claim_or_poll"], loc, 10, 18),
            _range(index["task_handoff_sync"], loc, 18, 20),
            _range(index["task_metadata"], loc, 20, 22),
            _range(index["mma_task"], loc, 22, 900, payload=0),
            _range(index["mma_slice"], loc, 23, 899, payload=0),
            _range(index["fc1_gate"], loc, 30, gate_end, payload=0),
            _range(index["qmma"], loc, 40, 50, payload=_phase_payload("gate", 0)),
            _range(index["fc1_up"], loc, up_start, up_end, payload=0),
            _range(
                index["qmma"],
                loc,
                up_start + 10,
                up_start + 20,
                payload=_phase_payload("up", 0),
            ),
            _range(index["act_quant"], loc, act_start, act_end, payload=0),
            _range(index["fc2_block"], loc, fc2_start, fc2_end, payload=0),
            _range(
                index["qmma"],
                loc,
                fc2_start + 10,
                fc2_start + 20,
                payload=_phase_payload("fc2", 0),
            ),
        ]
    )
    if long_wait:
        # One rare wait dominates ten short planned leaves by duration.
        ranges.append(
            _range(
                index["wait"],
                loc,
                60,
                660,
                payload=_phase_payload("gate", 1),
            )
        )
        for ordinal in range(10):
            start = 661 + ordinal * 2
            ranges.append(
                _range(
                    index["s2r"],
                    loc,
                    start,
                    start + 1,
                    payload=_phase_payload("gate", ordinal),
                )
            )
    return ranges


def _workspace(task_count: int = 1) -> dict:
    tasks = [
        {
            "task_slot": slot,
            "expert": 0,
            "m_tile": slot,
            "slice_begin": 0,
            "slice_count": 1,
            "valid_rows": 128,
            "ready": 0,
        }
        for slot in range(task_count)
    ]
    return {
        "expected_task_count": task_count,
        "task_head": task_count + 110,
        "task_tail": task_count,
        "row_counts_sum": 8192 * 8,
        "task_model_pass": True,
        "task_table": tasks,
        "hashes": {"task_table": "stable"},
    }


def make_capture(
    root: Path,
    *,
    pid: int = 1234,
    capture_id: str = "capture-0",
    graph_nodes: int = 1,
    regular_only: bool = False,
    overlap: bool = False,
    long_wait: bool = False,
    raw_bytes_delta: int = 0,
    raw_word_count: int = 4,
    qmma_model: dict | None = None,
    pc_gate: bool = True,
    candidate_hash: str = "candidate",
) -> Path:
    names = [
        "moe_kernel",
        "marker_calibration",
        "phase0_init",
        "histogram",
        "prefix_sum",
        "route_pack",
        "setup_compute",
        "task_claim_or_poll",
        "task_handoff_sync",
        "task_metadata",
        "mma_task",
        "mma_slice",
        "fc1_gate",
        "qmma",
        "fc1_up",
        "act_quant",
        "fc2_block",
        "wait",
        "s2r",
    ]
    ranges = []
    for warp in range(5):
        ranges.extend(_task_ranges(names, warp, long_wait=long_wait))
    if overlap:
        # Crosses the end of mma_task [22,900] on warp 0.
        ranges.append(_range(names.index("act_quant"), 0, 850, 950, payload=9))
    launch = {
        "kernelName": KERNEL,
        "contextId": 7,
        "gridId": 42,
        "gridDimX": 1,
        "gridDimY": 1,
        "gridDimZ": 110,
        "blockDimX": 160,
        "blockDimY": 1,
        "blockDimZ": 1,
        "ranges": ranges,
        "markers": [],
        "warpLifetimes": [],
    }
    graph = (
        {}
        if regular_only
        else {"graph-key": [dict(launch) for _ in range(graph_nodes)]}
    )
    decoded = {
        "stringTable": names,
        "locationTable": [_location(warp) for warp in range(5)],
        "launches": [launch] if regular_only else [],
        "graphLaunches": graph,
    }
    pid_dir = root / "iket" / f"pid_0x{pid:x}"
    pid_dir.mkdir(parents=True)
    (pid_dir / "iket.decoded_results.json").write_text(json.dumps(decoded))

    raw_launch = {
        key: value
        for key, value in launch.items()
        if key not in {"ranges", "markers", "warpLifetimes"}
    }
    raw_launch["maxTsCntPerWarp"] = 100
    raw_launch["warps"] = []
    for warp in range(5):
        raw_data = list(range(raw_word_count))
        raw_launch["warps"].append(
            {
                "ctaId": 0,
                "warpId": warp,
                "buffer": {
                    "header": {
                        "bytesWritten": 16 + len(raw_data) * 4 + raw_bytes_delta,
                    },
                    "raw_data": raw_data,
                },
            }
        )
    raw = {
        "launches": [raw_launch] if regular_only else [],
        "graphLaunches": ({} if regular_only else {"graph-key": [raw_launch]}),
    }
    (pid_dir / "iket.data.json").write_text(json.dumps(raw))

    config_dir = root / "gen-config"
    config_dir.mkdir()
    config_path = config_dir / "instrument.config.json"
    config_path.write_text(
        json.dumps(
            {
                "configs": [
                    {
                        "kernel": KERNEL,
                        "maxTsCntPerWarp": 100,
                        "instrumentations": [],
                    }
                ]
            }
        )
    )
    manifest_dir = root / "target-manifests" / f"pid_{pid}"
    manifest_dir.mkdir(parents=True)
    evidence = {
        "qmma_static_count_per_warp_slice": qmma_model,
        "fc2_blocks_per_slice": 1,
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "pid": pid,
        "trace_overflow": None,
        "capture": {
            "capture_id": capture_id,
            "cluster_id": capture_id,
            "selected_cta": [0, 0, 0],
        },
        "graph": {
            "kernel_name_pattern": "MoEDynamicKernel",
            "expected_grid": [1, 1, 110],
            "expected_block": [160, 1, 1],
            "context_id": None,
            "graph_launch_key": None,
            "grid_id": None,
        },
        "workspace": _workspace(),
        "instrumentation_evidence": evidence,
        "identity": {
            "candidate_manifest_sha256": candidate_hash,
            "jit_before": {"jit_artifact_set_sha256": "jit"},
            "jit_after": {"jit_artifact_set_sha256": "jit"},
        },
    }
    (manifest_dir / "target_manifest.json").write_text(json.dumps(manifest))
    tracker_dir = root / "tracker" / f"pid_0x{pid:x}"
    tracker_dir.mkdir(parents=True)
    tracker_cubin = tracker_dir / "module_0.cubin"
    tracker_cubin.write_bytes(b"synthetic exp003 tracker cubin")
    if pc_gate:
        (root / "pc_sass_gate.json").write_text(
            json.dumps(
                {
                    "schema_version": "exp003.pc_sass_gate.v1",
                    "status": "pass",
                    "fail_closed": True,
                    "artifacts": {
                        "tracker_cubin": {
                            "sha256": _file_sha256(tracker_cubin),
                        },
                        "instrument_config": {
                            "sha256": _file_sha256(config_path),
                        },
                    },
                    "verified_range_names": ["fc1_gate_wait"],
                    "overall_pass": True,
                }
            )
        )
    return root


def write_binary_gate(
    root: Path,
    *,
    static_omma_count: int = 896,
    eligible: bool = True,
    reasons: list[str] | None = None,
) -> Path:
    tracker_cubin = next((root / "tracker").rglob("*.cubin"))
    path = root / "binary_gate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate": {"cubin_sha256": _file_sha256(tracker_cubin)},
                "binary_semantic_omma_gate": {
                    "pass": True,
                    "control_static_semantic_omma_count": static_omma_count,
                    "candidate_static_semantic_omma_count": static_omma_count,
                    "reason": None,
                },
                "formal_dominance": {
                    "eligible": eligible,
                    "fail_closed": True,
                    "reasons": [] if reasons is None else reasons,
                },
            }
        )
    )
    return path


def cadence_gap(phase: str, warp: int, duration: int) -> dict:
    return {
        "phase": phase,
        "warp_id": warp,
        "duration": duration,
        "tensor_duration": duration,
        "planned_duration": 0,
        "starvation_duration": 0,
        "orchestration_duration": 0,
        "unclassified_duration": 0,
    }


class CaptureParsingTests(unittest.TestCase):
    def test_overlay_matches_30_event_phase_payload_contract(self):
        overlay = (
            MODULE_PATH.parent
            / "overlays/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel_iket.py"
        )
        source = overlay.read_text()
        names = set(
            re.findall(r'range_(?:push|start)\(\s*"([^"]+)"', source, re.MULTILINE)
        )
        self.assertLessEqual(len(names), 30)
        self.assertTrue({"qmma", "s2r", "wait", "tma_acquire", "tma_issue"} <= names)
        self.assertFalse(names & analyzer.LEGACY_SPLIT_RANGE_NAMES)
        self.assertIn("_IKET_PHASE_STRIDE = 1_000_000", source)
        self.assertIn("_IKET_PHASE_GATE = 1", source)
        self.assertIn("_IKET_PHASE_UP = 2", source)
        self.assertIn("_IKET_PHASE_FC2 = 3", source)

    def test_unique_graph_node_ignores_no_regular_eager_and_binds_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp))
            decoded_path = next(root.rglob("iket.decoded_results.json"))
            decoded = json.loads(decoded_path.read_text())
            decoded["launches"].append(dict(decoded["graphLaunches"]["graph-key"][0]))
            decoded_path.write_text(json.dumps(decoded))
            capture = analyzer.load_capture(root)
            self.assertEqual(capture.graph_key, "graph-key")
            self.assertEqual(capture.context_id, 7)
            self.assertEqual(capture.grid_id, 42)
            self.assertEqual(capture.pid, 1234)
            self.assertFalse(capture.trace_capacity["trace_overflow"])

    def test_large_raw_timestamps_remain_exact_integers(self):
        base = 2**60
        document = {
            "stringTable": ["exact"],
            "locationTable": [_location(0)],
        }
        launch = {
            "ranges": [_range(0, 0, base, base + 32)],
        }
        record = analyzer._parse_ranges(document, launch)[0]
        self.assertEqual(record.duration, 32)
        self.assertIsInstance(record.start, int)

    def test_regular_eager_only_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp), regular_only=True)
            with self.assertRaisesRegex(analyzer.AnalysisError, "regular/eager"):
                analyzer.load_capture(root)

    def test_multiple_matching_graph_nodes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp), graph_nodes=2)
            with self.assertRaisesRegex(analyzer.AnalysisError, "exactly one"):
                analyzer.load_capture(root)

    def test_start_event_payload_and_nested_hierarchy_are_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = analyzer.load_capture(make_capture(Path(tmp)))
            qmma = next(
                record for record in capture.ranges if record.name == "fc1_gate_qmma"
            )
            self.assertEqual(qmma.payload, 0)
            self.assertEqual(qmma.raw_name, "qmma")
            self.assertEqual(qmma.raw_payload, _phase_payload("gate", 0))
            self.assertEqual(qmma.phase, "gate")
            by_index = {record.index: record for record in capture.ranges}
            self.assertEqual(
                [item.name for item in analyzer._ancestors(qmma, by_index)[:4]],
                ["fc1_gate", "mma_slice", "mma_task", "moe_kernel"],
            )

    def test_phase_payload_must_match_range_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp))
            decoded_path = next(root.rglob("iket.decoded_results.json"))
            decoded = json.loads(decoded_path.read_text())
            launch = decoded["graphLaunches"]["graph-key"][0]
            qmma_index = decoded["stringTable"].index("qmma")
            gate_payload = _phase_payload("gate", 0)
            target = next(
                row
                for row in launch["ranges"]
                if row["rangeNameIdx"] == qmma_index
                and row["internalEvents"][0].get("payloadVal") == gate_payload
            )
            target["internalEvents"][0]["payloadVal"] = _phase_payload("up", 0)
            decoded_path.write_text(json.dumps(decoded))
            with self.assertRaisesRegex(
                analyzer.AnalysisError, "not nested under fc1_up"
            ):
                analyzer.load_capture(root)

    def test_zero_duration_qmma_at_inclusive_parent_end_is_reparented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp))
            decoded_path = next(root.rglob("iket.decoded_results.json"))
            decoded = json.loads(decoded_path.read_text())
            launch = decoded["graphLaunches"]["graph-key"][0]
            names = decoded["stringTable"]
            up_payload = _phase_payload("up", 0)
            qmma = next(
                row
                for row in launch["ranges"]
                if names[row["rangeNameIdx"]] == "qmma"
                and row["internalEvents"][0].get("payloadVal") == up_payload
                and row["warpLocIdxs"][0] == 0
            )
            qmma["startTs"] = qmma["endTs"] = 180
            for event in qmma["internalEvents"]:
                event["timestamp"] = 180
            act_quant = next(
                row
                for row in launch["ranges"]
                if names[row["rangeNameIdx"]] == "act_quant"
                and row["warpLocIdxs"][0] == 0
            )
            act_quant["startTs"] = 180
            act_quant["internalEvents"][0]["timestamp"] = 180
            decoded_path.write_text(json.dumps(decoded))

            capture = analyzer.load_capture(root)
            record = next(
                row
                for row in capture.ranges
                if row.name == "fc1_up_qmma" and row.location.warp == 0
            )
            by_index = {row.index: row for row in capture.ranges}
            self.assertEqual(record.duration, 0)
            self.assertEqual(analyzer._ancestors(record, by_index)[0].name, "fc1_up")

    def test_provider_user_event_name_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp))
            decoded_path = next(root.rglob("iket.decoded_results.json"))
            decoded = json.loads(decoded_path.read_text())
            launch = decoded["graphLaunches"]["graph-key"][0]
            used = {
                decoded["stringTable"][row["rangeNameIdx"]] for row in launch["ranges"]
            }
            for ordinal in range(31 - len(used)):
                name_index = len(decoded["stringTable"])
                decoded["stringTable"].append(f"extra_{ordinal}")
                launch["ranges"].append(
                    _range(name_index, 4, 900 + ordinal * 2, 901 + ordinal * 2)
                )
            decoded_path.write_text(json.dumps(decoded))
            with self.assertRaisesRegex(analyzer.AnalysisError, "at most 30"):
                analyzer.load_capture(root)

    def test_partial_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp), overlap=True)
            with self.assertRaisesRegex(
                analyzer.AnalysisError, "partially overlapping"
            ):
                analyzer.load_capture(root)

    def test_native_dump_header_closure_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp), raw_bytes_delta=-16)
            with self.assertRaisesRegex(analyzer.AnalysisError, "16-byte header"):
                analyzer.load_capture(root)

    def test_native_dump_requires_ten_percent_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp), raw_word_count=180)
            with self.assertRaisesRegex(analyzer.AnalysisError, "10% headroom"):
                analyzer.load_capture(root)

    def test_candidate_identity_drift_across_captures_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_capture(root / "first", capture_id="capture-0")
            second = make_capture(
                root / "second",
                capture_id="capture-1",
                candidate_hash="different-candidate",
            )
            with self.assertRaisesRegex(analyzer.AnalysisError, "identity drift"):
                analyzer.analyze_experiment([first, second], bootstrap_replicates=10)

    def test_ephemeral_container_hostname_is_not_runtime_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = analyzer.load_capture(
                make_capture(root / "first", capture_id="capture-0")
            )
            second = analyzer.load_capture(
                make_capture(root / "second", capture_id="capture-1")
            )
            first.manifest["runtime"] = {
                "hostname": "ephemeral-container-a",
                "gpu": {"visible_uuid": "GPU-stable"},
            }
            second.manifest["runtime"] = {
                "hostname": "ephemeral-container-b",
                "gpu": {"visible_uuid": "GPU-stable"},
            }
            analyzer._identity_drift_gate([first, second])
            second.manifest["runtime"]["gpu"]["visible_uuid"] = "GPU-drift"
            with self.assertRaisesRegex(analyzer.AnalysisError, "runtime.*drift"):
                analyzer._identity_drift_gate([first, second])

    def test_null_qmma_model_is_not_filled_with_guessed_32(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = analyzer.load_capture(make_capture(Path(tmp), qmma_model=None))
            self.assertIsNone(capture.event_model["fc1_gate_qmma"])
            result = analyzer.analyze_experiment(
                [capture.root], bootstrap_replicates=10
            )
            self.assertEqual(
                result["event_count_evidence"]["resolved_counts_per_warp_slice"][
                    "fc1_gate_qmma"
                ],
                1,
            )
            self.assertFalse(
                result["event_count_evidence"]["formal_event_count_closure"]
            )
            self.assertEqual(
                result["weighted_phase_shares"]["decision"], "inconclusive"
            )
            self.assertEqual(len(result["top_level_phase_unions"]), 25)
            output = Path(tmp) / "derived"
            analyzer.write_outputs(result, output)
            self.assertTrue((output / "weighted_phase_shares.json").is_file())
            self.assertTrue((output / "qmma_gaps.csv").is_file())
            self.assertTrue((output / "cadence_summary.json").is_file())
            self.assertTrue((output / "top_level_phase_unions.csv").is_file())

    def test_static_omma_count_is_not_compared_to_dynamic_event_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(
                Path(tmp),
                qmma_model={
                    "fc1_gate_qmma": 1,
                    "fc1_up_qmma": 1,
                    "fc2_qmma": 1,
                },
            )
            binary_gate = write_binary_gate(root, static_omma_count=896)
            result = analyzer.analyze_experiment(
                [root], binary_gate_path=binary_gate, bootstrap_replicates=10
            )
            evidence = result["event_count_evidence"]
            self.assertEqual(evidence["runtime_event_total_per_warp_slice"], 3)
            self.assertEqual(evidence["candidate_static_semantic_omma_count"], 896)
            self.assertIn("forbidden", evidence["static_runtime_comparison_policy"])
            self.assertTrue(evidence["formal_event_count_closure"])

    def test_binary_omma_gate_can_close_independently_of_runtime_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(
                Path(tmp),
                qmma_model={
                    "fc1_gate_qmma": 1,
                    "fc1_up_qmma": 1,
                    "fc2_qmma": 1,
                },
            )
            binary_gate = write_binary_gate(root, static_omma_count=896)
            result = analyzer.analyze_experiment(
                [root], binary_gate_path=binary_gate, bootstrap_replicates=10
            )
            evidence = result["event_count_evidence"]
            self.assertEqual(evidence["runtime_event_total_per_warp_slice"], 3)
            self.assertTrue(evidence["binary_semantic_omma_gate_pass"])
            self.assertTrue(evidence["formal_event_count_closure"])

    def test_formal_dominance_is_separate_from_event_count_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(
                Path(tmp),
                qmma_model={
                    "fc1_gate_qmma": 1,
                    "fc1_up_qmma": 1,
                    "fc2_qmma": 1,
                },
            )
            binary_gate = write_binary_gate(
                root,
                static_omma_count=896,
                eligible=False,
                reasons=["resource identity failed: STACK"],
            )
            result = analyzer.analyze_experiment(
                [root], binary_gate_path=binary_gate, bootstrap_replicates=10
            )
            self.assertTrue(
                result["event_count_evidence"]["formal_event_count_closure"]
            )
            weighted = result["weighted_phase_shares"]
            self.assertFalse(weighted["formal_dominance_eligible"])
            self.assertEqual(weighted["decision"], "inconclusive")
            self.assertIn(
                "resource identity failed: STACK", weighted["decision_reasons"]
            )

    def test_missing_pc_sass_gate_is_diagnostic_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = analyzer.load_capture(make_capture(Path(tmp), pc_gate=False))
            self.assertEqual(capture.pc_sass_verified_ranges, frozenset())
            self.assertFalse(capture.pc_sass_gate["pass"])
            self.assertIn("missing", capture.pc_sass_gate["reason"])

    def test_pc_sass_gate_tracker_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(Path(tmp))
            gate_path = root / "pc_sass_gate.json"
            gate = json.loads(gate_path.read_text())
            gate["artifacts"]["tracker_cubin"]["sha256"] = "0" * 64
            gate_path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(analyzer.AnalysisError, "tracker cubin SHA"):
                analyzer.load_capture(root)


class CadenceSummaryTests(unittest.TestCase):
    def test_phase_percentiles_thresholds_and_bucket_totals(self):
        calibration = {str(warp): {"p95": 1} for warp in range(4)}
        rows = [
            cadence_gap("fc1_gate_qmma", warp, duration)
            for warp, duration in zip((0, 1, 2, 3, 0), (1, 2, 3, 4, 5), strict=True)
        ]
        summary = analyzer._cadence_summary(rows, calibration)

        self.assertEqual(summary["schema_version"], "exp003.cadence_summary.v1")
        self.assertEqual(summary["timestamp_unit"], "raw timestamp units")
        self.assertEqual(summary["calibration_p95"]["mode"], "consistent")
        self.assertEqual(summary["calibration_p95"]["value"], 1)
        phase = summary["phases"]["fc1_gate_qmma"]
        self.assertEqual(phase["count"], 5)
        self.assertEqual(phase["gap_count_per_warp"], {"0": 2, "1": 1, "2": 1, "3": 1})
        self.assertEqual(phase["duration"]["p50"], 3)
        self.assertAlmostEqual(phase["duration"]["p95"], 4.8)
        self.assertAlmostEqual(phase["duration"]["p99"], 4.96)
        self.assertAlmostEqual(phase["duration"]["p99.9"], 4.996)
        self.assertEqual(phase["duration"]["max"], 5)
        self.assertEqual(
            phase["exceedance"]["gt_calibration_p95"],
            {"count": 4, "percentage": 80.0},
        )
        self.assertEqual(
            phase["exceedance"]["gt_2x_calibration_p95"],
            {"count": 3, "percentage": 60.0},
        )
        self.assertEqual(
            phase["exceedance"]["gt_4x_calibration_p95"],
            {"count": 1, "percentage": 20.0},
        )
        self.assertEqual(phase["total_gap_duration"], 15)
        self.assertEqual(phase["bucket_duration_totals"]["tensor"], 15)

        empty = summary["phases"]["fc2_qmma"]
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["duration"]["p99.9"])
        self.assertIsNone(empty["exceedance"]["gt_calibration_p95"]["percentage"])

    def test_per_warp_calibration_is_used_for_thresholds(self):
        calibration = {str(warp): {"p95": p95} for warp, p95 in enumerate((1, 2, 3, 4))}
        rows = [
            cadence_gap("fc1_up_qmma", warp, duration)
            for warp, duration in enumerate((2, 3, 4, 5))
        ]
        summary = analyzer._cadence_summary(rows, calibration)

        self.assertEqual(summary["calibration_p95"]["mode"], "per_warp")
        self.assertIsNone(summary["calibration_p95"]["value"])
        self.assertEqual(
            summary["calibration_p95"]["per_warp"],
            {"0": 1, "1": 2, "2": 3, "3": 4},
        )
        phase = summary["phases"]["fc1_up_qmma"]
        self.assertEqual(
            phase["exceedance"]["gt_calibration_p95"],
            {"count": 4, "percentage": 100.0},
        )
        self.assertEqual(
            phase["exceedance"]["gt_2x_calibration_p95"],
            {"count": 0, "percentage": 0.0},
        )


class IntervalAndDecisionTests(unittest.TestCase):
    def test_rare_long_wait_beats_many_short_planned_ranges_by_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_capture(
                Path(tmp),
                long_wait=True,
                qmma_model={
                    "fc1_gate_qmma": 1,
                    "fc1_up_qmma": 1,
                    "fc2_qmma": 1,
                },
            )
            result = analyzer.analyze_experiment([root], bootstrap_replicates=10)
            observation = result["observations"][0]
            planned_leaf_count = sum(
                1
                for row in result["leaf_intervals"]
                if row["task_slot"] == 0
                and row["warp_id"] == 0
                and row["bucket"] == "planned"
            )
            self.assertGreater(planned_leaf_count, 5)
            self.assertGreater(
                observation["durations"]["starvation"],
                observation["durations"]["planned"],
            )

    def test_mixed_and_unclassified_decisions(self):
        intervals = {
            "tensor": {"lower": 0.05, "upper": 0.10},
            "planned": {"lower": 0.25, "upper": 0.35},
            "starvation": {"lower": 0.22, "upper": 0.32},
            "orchestration": {"lower": 0.10, "upper": 0.15},
            "unclassified": {"lower": 0.05, "upper": 0.10},
        }
        self.assertEqual(analyzer.decide(True, intervals)[0], "mixed")
        intervals["unclassified"] = {"lower": 0.10, "upper": 0.21}
        self.assertEqual(analyzer.decide(True, intervals)[0], "inconclusive")

    def test_coverage_requires_eight_tasks_and_three_captures_per_stratum(self):
        population = {
            "strata": [
                {"stratum": "early|full|slices=1", "population_tasks": 8},
                {"stratum": "steady|full|slices=1", "population_tasks": 8},
                {"stratum": "tail|full|slices=1", "population_tasks": 8},
            ]
        }
        rows = []
        for stratum in ("early", "steady", "tail"):
            for index in range(8):
                rows.append(
                    {
                        "stratum": f"{stratum}|full|slices=1",
                        "capture_id": f"capture-{index % 3}",
                    }
                )
        self.assertTrue(analyzer._coverage(rows, population)["pass"])
        two_capture_rows = [
            {**row, "capture_id": f"capture-{index % 2}"}
            for index, row in enumerate(rows)
        ]
        self.assertFalse(analyzer._coverage(two_capture_rows, population)["pass"])

    def test_capture_cluster_bootstrap_is_seeded_and_can_decide_dominance(self):
        population = {
            "strata": [
                {
                    "stratum": "steady|full|slices=1",
                    "population_tasks": 24,
                    "population_weight": 1.0,
                }
            ]
        }
        observations = []
        for capture in range(3):
            for task in range(8):
                for warp in range(4):
                    observations.append(
                        {
                            "capture_id": f"capture-{capture}",
                            "task_slot": capture * 8 + task,
                            "stratum": "steady|full|slices=1",
                            "warp_id": warp,
                            "complete_task": True,
                            "event_count_closure": {"pass": True},
                            "shares": {
                                "tensor": 0.15,
                                "planned": 0.65,
                                "starvation": 0.05,
                                "orchestration": 0.10,
                                "unclassified": 0.05,
                            },
                        }
                    )
        first = analyzer.aggregate(observations, population, bootstrap_replicates=300)
        second = analyzer.aggregate(observations, population, bootstrap_replicates=300)
        self.assertEqual(first["bootstrap"], second["bootstrap"])
        self.assertEqual(first["decision"], "planned-dominant")


if __name__ == "__main__":
    unittest.main()
