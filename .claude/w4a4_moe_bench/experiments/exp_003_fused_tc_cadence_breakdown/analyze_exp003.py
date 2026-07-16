#!/usr/bin/env python3
"""Assignment-aware IKET analysis for exp_003.

The analyzer deliberately keeps IKET timestamps in raw units.  It selects one
CUDA-Graph replay node per capture, joins it to the target manifest written by
the *same PID*, and treats every MMA warp as an independent timeline.  It does
not turn instrumented durations into production latency or partition NCU's
whole-launch counters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


TIMESTAMP_UNIT = "raw timestamp units"
BUCKETS = ("tensor", "planned", "starvation", "orchestration", "unclassified")
DECISION_BUCKETS = ("planned", "starvation", "orchestration")
MMA_WARPS = frozenset((0, 1, 2, 3))
TILE_M = 128
FIXED_ROW_COUNT = 8192 * 8
BOOTSTRAP_SEED = 20260716
MAX_USER_EVENT_NAMES = 30
PAYLOAD_PHASE_STRIDE = 1_000_000
PAYLOAD_PHASES = {1: "gate", 2: "up", 3: "fc2"}
MERGED_RANGE_NAMES = frozenset({"qmma", "s2r", "wait", "tma_acquire", "tma_issue"})
LEGACY_SPLIT_RANGE_NAMES = frozenset(
    {
        f"{prefix}_{kind}"
        for prefix in ("fc1_gate", "fc1_up", "fc2")
        for kind in ("qmma", "s2r", "wait")
    }
    | {
        f"tma_{phase}_{kind}"
        for phase in ("gate", "up", "fc2")
        for kind in ("acquire", "issue")
    }
)

# Source geometry proves 16 FC2 output blocks, but it does *not* prove the
# number of dynamic QMMA events seen by one warp.  Those counts are learned
# from closed decoded ranges and, for formal use, cross-checked against the
# same instrumented cubin.  Never reintroduce guessed 32/32/32 defaults here.
DEFAULT_EVENT_MODEL: dict[str, int | None] = {
    "fc1_gate_qmma": None,
    "fc1_up_qmma": None,
    "fc2_qmma": None,
    "fc2_blocks_per_slice": 16,
}

PLANNED_LEAVES = frozenset(
    {
        "phase0_init",
        "histogram",
        "prefix_sum",
        "route_pack",
        "setup_compute",
        "fc1_gate_s2r",
        "fc1_up_s2r",
        "act_quant",
        "fc2_s2r",
        "fc2_epilogue",
        "fc2_atomic_scatter",
        "tma_gate_acquire",
        "tma_gate_issue",
        "tma_up_acquire",
        "tma_up_issue",
        "tma_fc2_acquire",
        "tma_fc2_issue",
    }
)
WAIT_LEAVES = frozenset(
    {
        "fc1_gate_wait",
        "fc1_up_wait",
        "fc2_wait",
        "fc2_pre_scatter_barrier",
        "fc2_post_scatter_barrier",
        "gate_pass_wait",
        "final_pass_wait",
    }
)
ORCHESTRATION_LEAVES = frozenset(
    {
        "task_claim_or_poll",
        "task_handoff_sync",
        "task_metadata",
        "task_tail_exit",
    }
)
REQUIRED_TASK_PHASES = ("fc1_gate_qmma", "fc1_up_qmma", "fc2_qmma")
TOP_LEVEL_PHASES = (
    "phase0_init",
    "histogram",
    "prefix_sum",
    "route_pack",
    "setup_compute",
)


class AnalysisError(ValueError):
    """A failed evidence gate, not a zero-valued measurement."""


def _finite_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise AnalysisError(f"{label} must be finite")
    # IKET timestamps are commonly >2**53.  Preserve JSON integers exactly;
    # converting them to float would erase short ranges and QMMA gaps.
    return value


def _integer(value: Any, label: str) -> int:
    number = _finite_number(value, label)
    if isinstance(number, float) and not number.is_integer():
        raise AnalysisError(f"{label} must be an integer")
    return int(number)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[tuple[float, float]] = []
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return merged


def _duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _union(intervals))


def _intersection_duration(
    intervals: Iterable[tuple[float, float]], start: float, end: float
) -> float:
    return _duration(
        (max(start, left), min(end, right))
        for left, right in intervals
        if min(end, right) > max(start, left)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            f"{path}: invalid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: top-level JSON must be an object")
    return value


def _normalize_pid(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} is not a PID")
    if isinstance(value, int):
        if value < 0:
            raise AnalysisError(f"{label} is negative")
        return value
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{label} is not a PID")
    text = value.strip().lower()
    if text.startswith("pid_") or text.startswith("pid-"):
        text = text[4:]
    base = 16 if text.startswith("0x") else 10
    try:
        result = int(text, base)
    except ValueError as exc:
        raise AnalysisError(f"{label}={value!r} is not a PID") from exc
    if result < 0:
        raise AnalysisError(f"{label} is negative")
    return result


def _pid_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"pid[_-]((?:0x)?[0-9a-fA-F]+)", part)
        if match:
            return _normalize_pid(match.group(1), f"PID in {path}")
    raise AnalysisError(f"cannot recover PID from decoded path: {path}")


def _as_xyz(value: Any, label: str) -> tuple[int, int, int]:
    if isinstance(value, str):
        pieces = value.split(",")
        if len(pieces) == 3:
            value = pieces
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AnalysisError(f"{label} must contain x,y,z")
    return tuple(_integer(item, label) for item in value)  # type: ignore[return-value]


@dataclass(frozen=True, order=True)
class WarpLocation:
    gpc: int
    tpc: int
    sm: int
    cta_x: int
    cta_y: int
    cta_z: int
    warp: int

    @property
    def cta(self) -> tuple[int, int, int]:
        return (self.cta_x, self.cta_y, self.cta_z)

    @property
    def key(self) -> str:
        return (
            f"gpc{self.gpc}/tpc{self.tpc}/sm{self.sm}/"
            f"cta({self.cta_x},{self.cta_y},{self.cta_z})/warp{self.warp}"
        )


@dataclass
class RangeRecord:
    index: int
    name: str
    raw_name: str
    start: float
    end: float
    payload: int | float | None
    raw_payload: int | float | None
    phase: str | None
    location: WarpLocation
    scope: int
    parent: int | None = None
    children: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Capture:
    root: Path
    decoded_path: Path
    manifest_path: Path
    pid: int
    capture_id: str
    cluster_id: str
    selected_cta: tuple[int, int, int]
    graph_key: str
    context_id: int
    grid_id: int
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    user_event_names: tuple[str, ...]
    ranges: list[RangeRecord]
    manifest: dict[str, Any]
    event_model: dict[str, int | None]
    pc_sass_verified_ranges: frozenset[str]
    pc_sass_gate: dict[str, Any]
    tracker_cubin_path: Path
    tracker_cubin_sha256: str
    trace_capacity: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "pid": self.pid,
            "context_id": self.context_id,
            "graph_launch_key": self.graph_key,
            "grid_id": self.grid_id,
            "kernel_name": self.kernel_name,
            "grid": list(self.grid),
            "block": list(self.block),
            "user_event_name_count": len(self.user_event_names),
            "user_event_names": list(self.user_event_names),
            "selected_cta": list(self.selected_cta),
            "decoded_path": str(self.decoded_path),
            "decoded_sha256": _sha256(self.decoded_path),
            "target_manifest": str(self.manifest_path),
            "target_manifest_sha256": _sha256(self.manifest_path),
            "tracker_cubin": str(self.tracker_cubin_path),
            "tracker_cubin_sha256": self.tracker_cubin_sha256,
            "pc_sass_gate": self.pc_sass_gate,
            "trace_capacity": self.trace_capacity,
        }


def _resolve_string(document: Mapping[str, Any], index: Any, label: str) -> str:
    idx = _integer(index, label)
    table = document.get("stringTable")
    if not isinstance(table, list) or not 0 <= idx < len(table):
        raise AnalysisError(f"{label}={idx} is outside stringTable")
    value = table[idx]
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"stringTable[{idx}] is not a non-empty string")
    return value


def _range_name(document: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    if "rangeNameIdx" in record:
        return _resolve_string(document, record["rangeNameIdx"], "rangeNameIdx")
    name = record.get("rangeName")
    if not isinstance(name, str) or not name:
        raise AnalysisError("range is missing rangeName/rangeNameIdx")
    return name


def _location_from_value(value: Any, label: str) -> WarpLocation:
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be an object")
    cta = _as_xyz(value.get("ctaId"), f"{label}.ctaId")
    return WarpLocation(
        gpc=_integer(value.get("gpcId"), f"{label}.gpcId"),
        tpc=_integer(value.get("tpcId"), f"{label}.tpcId"),
        sm=_integer(value.get("smId"), f"{label}.smId"),
        cta_x=cta[0],
        cta_y=cta[1],
        cta_z=cta[2],
        warp=_integer(value.get("warpId"), f"{label}.warpId"),
    )


def _event_locations(
    document: Mapping[str, Any], record: Mapping[str, Any], event_count: int
) -> list[WarpLocation]:
    if "warpLocIdxs" in record:
        indices = record["warpLocIdxs"]
        if not isinstance(indices, list):
            raise AnalysisError("warpLocIdxs must be an array")
        table = document.get("locationTable")
        if not isinstance(table, list):
            raise AnalysisError("locationTable must be an array")
        locations = []
        for raw_index in indices:
            index = _integer(raw_index, "warpLocIdx")
            if not 0 <= index < len(table):
                raise AnalysisError(f"warpLocIdx={index} is outside locationTable")
            locations.append(
                _location_from_value(table[index], f"locationTable[{index}]")
            )
    else:
        values = record.get("warpLocs")
        if not isinstance(values, list):
            raise AnalysisError("range is missing warpLocIdxs/warpLocs")
        locations = [
            _location_from_value(value, f"warpLocs[{index}]")
            for index, value in enumerate(values)
        ]
    if event_count and len(locations) != event_count:
        raise AnalysisError(
            f"range has {event_count} internal events but {len(locations)} locations"
        )
    if not locations:
        raise AnalysisError("range has no event location")
    if len(set(locations)) != 1:
        raise AnalysisError("range crosses warp locations")
    return locations


def _parse_ranges(
    document: Mapping[str, Any], launch: Mapping[str, Any]
) -> list[RangeRecord]:
    raw_ranges = launch.get("ranges")
    if not isinstance(raw_ranges, list):
        raise AnalysisError("target graph node ranges must be an array")
    result: list[RangeRecord] = []
    for index, value in enumerate(raw_ranges):
        if not isinstance(value, dict):
            raise AnalysisError(f"range[{index}] must be an object")
        name = _range_name(document, value)
        start = _finite_number(value.get("startTs"), f"range[{index}].startTs")
        end = _finite_number(value.get("endTs"), f"range[{index}].endTs")
        if end < start:
            raise AnalysisError(f"range[{index}] {name}: endTs < startTs")
        events = value.get("internalEvents")
        if not isinstance(events, list) or not events:
            raise AnalysisError(f"range[{index}] {name}: missing start/end events")
        locations = _event_locations(document, value, len(events))
        event_timestamps: list[float] = []
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                raise AnalysisError(
                    f"range[{index}] event[{event_index}] is not an object"
                )
            timestamp = _finite_number(
                event.get("timestamp"), f"range[{index}] event[{event_index}].timestamp"
            )
            if timestamp < start or timestamp > end:
                raise AnalysisError(
                    f"range[{index}] {name}: event outside range bounds"
                )
            event_timestamps.append(timestamp)
        if event_timestamps != sorted(event_timestamps):
            raise AnalysisError(f"range[{index}] {name}: events are out of order")
        if event_timestamps[0] != start:
            raise AnalysisError(
                f"range[{index}] {name}: first event is not range start"
            )
        if event_timestamps[-1] != end:
            raise AnalysisError(f"range[{index}] {name}: last event is not range end")
        start_event = events[0]
        payload: int | float | None = start_event.get("payloadVal")
        if payload is not None:
            payload_number = _finite_number(
                payload, f"range[{index}] {name} payloadVal"
            )
            payload = (
                int(payload_number)
                if isinstance(payload_number, int) or payload_number.is_integer()
                else payload_number
            )
        result.append(
            RangeRecord(
                index=index,
                name=name,
                raw_name=name,
                start=start,
                end=end,
                payload=payload,
                raw_payload=payload,
                phase=None,
                location=locations[0],
                scope=_integer(value.get("rangeScope"), f"range[{index}].rangeScope"),
            )
        )
    _build_nesting(result)
    return result


def _decode_phase_payload(payload: Any, label: str) -> tuple[str, int]:
    encoded = _integer(payload, label)
    if encoded < PAYLOAD_PHASE_STRIDE:
        raise AnalysisError(
            f"{label}={encoded} is not phase-coded with stride {PAYLOAD_PHASE_STRIDE}"
        )
    phase_id, ordinal_plus_one = divmod(encoded, PAYLOAD_PHASE_STRIDE)
    phase = PAYLOAD_PHASES.get(phase_id)
    if phase is None:
        raise AnalysisError(f"{label}={encoded} has unknown phase_id={phase_id}")
    return phase, ordinal_plus_one - 1


def _reparent_zero_duration_end_boundary(
    record: RangeRecord,
    required_parent_name: str,
    ranges: Sequence[RangeRecord],
    by_index: Mapping[int, RangeRecord],
) -> bool:
    """Resolve only an unambiguous inclusive-end boundary attachment.

    IKET may emit a zero-duration leaf at exactly the previous semantic
    parent's inclusive end and the next sibling's start.  Generic timestamp
    nesting then attaches it to the next sibling.  No non-zero or ambiguous
    interval is repaired here.
    """
    if record.duration != 0 or record.children:
        return False
    candidates = [
        candidate
        for candidate in ranges
        if candidate.index != record.index
        and candidate.location == record.location
        and candidate.name == required_parent_name
        and candidate.end == record.start == record.end
        and _contains(candidate, record)
    ]
    if len(candidates) != 1:
        return False
    candidate = candidates[0]
    if record.parent is not None:
        current_parent = by_index[record.parent]
        current_parent.children = [
            child for child in current_parent.children if child != record.index
        ]
    record.parent = candidate.index
    if record.index not in candidate.children:
        candidate.children.append(record.index)
    return True


def _decode_merged_ranges(ranges: Sequence[RangeRecord]) -> tuple[str, ...]:
    """Decode the provider-0.7.10 30-event payload contract fail-closed."""
    raw_names = tuple(sorted({record.raw_name for record in ranges}))
    if len(raw_names) > MAX_USER_EVENT_NAMES:
        raise AnalysisError(
            f"target uses {len(raw_names)} unique user range names; IKET 0.7.10 "
            f"supports at most {MAX_USER_EVENT_NAMES}"
        )
    legacy = sorted(set(raw_names) & LEGACY_SPLIT_RANGE_NAMES)
    if legacy:
        raise AnalysisError(
            "legacy split range names violate the 30-event contract: "
            + ", ".join(legacy)
        )
    if "qmma" not in raw_names:
        raise AnalysisError("target lacks the merged qmma range required by exp_003")

    by_index = {record.index: record for record in ranges}
    for record in ranges:
        if record.raw_name not in MERGED_RANGE_NAMES:
            continue
        phase, ordinal = _decode_phase_payload(
            record.raw_payload,
            f"{record.raw_name} range[{record.index}] payloadVal",
        )
        record.phase = phase
        record.payload = ordinal
        if record.raw_name in {"qmma", "s2r", "wait"}:
            prefix = {"gate": "fc1_gate", "up": "fc1_up", "fc2": "fc2"}[phase]
            record.name = f"{prefix}_{record.raw_name}"
            if record.location.warp not in MMA_WARPS:
                raise AnalysisError(
                    f"{record.raw_name} phase={phase} appears on non-MMA "
                    f"warp={record.location.warp}"
                )
            ancestor_names = {
                ancestor.name for ancestor in _ancestors(record, by_index)
            }
            required_ancestor = {"gate": "fc1_gate", "up": "fc1_up"}.get(phase)
            if phase == "fc2" and (
                record.raw_name in {"qmma", "wait"}
                or (record.raw_name == "s2r" and ordinal >= 0)
            ):
                required_ancestor = "fc2_block"
            if (
                required_ancestor is not None
                and required_ancestor not in ancestor_names
            ):
                _reparent_zero_duration_end_boundary(
                    record, required_ancestor, ranges, by_index
                )
                ancestor_names = {
                    ancestor.name for ancestor in _ancestors(record, by_index)
                }
                if required_ancestor not in ancestor_names:
                    raise AnalysisError(
                        f"{record.raw_name} phase={phase} is not nested under "
                        f"{required_ancestor}"
                    )
            if phase == "fc2" and "mma_slice" not in ancestor_names:
                raise AnalysisError(
                    f"{record.raw_name} phase=fc2 is not nested under mma_slice"
                )
            if record.raw_name in {"qmma", "wait"} and ordinal < 0:
                raise AnalysisError(
                    f"{record.raw_name} phase={phase} cannot use ordinal={ordinal}"
                )
            if (
                record.raw_name == "s2r"
                and ordinal < 0
                and not (phase == "fc2" and ordinal == -1)
            ):
                raise AnalysisError(
                    f"s2r phase={phase} has invalid sentinel ordinal={ordinal}"
                )
        else:
            tma_prefix = {"gate": "tma_gate", "up": "tma_up", "fc2": "tma_fc2"}[phase]
            suffix = "acquire" if record.raw_name == "tma_acquire" else "issue"
            record.name = f"{tma_prefix}_{suffix}"
            if record.location.warp != 4:
                raise AnalysisError(
                    f"{record.raw_name} phase={phase} appears on non-TMA "
                    f"warp={record.location.warp}"
                )
            if ordinal < 0:
                raise AnalysisError(
                    f"{record.raw_name} phase={phase} cannot use ordinal={ordinal}"
                )
            if "tma_slice" not in {
                ancestor.name for ancestor in _ancestors(record, by_index)
            }:
                raise AnalysisError(
                    f"{record.raw_name} phase={phase} is not nested under tma_slice"
                )
    return raw_names


def _contains(parent: RangeRecord, child: RangeRecord) -> bool:
    return parent.start <= child.start and child.end <= parent.end


def _build_nesting(ranges: list[RangeRecord]) -> None:
    """Build one containment tree per warp and reject crossing intervals."""
    by_warp: dict[WarpLocation, list[RangeRecord]] = defaultdict(list)
    for record in ranges:
        by_warp[record.location].append(record)
    for records in by_warp.values():
        ordered = sorted(records, key=lambda item: (item.start, -item.end, item.index))
        stack: list[RangeRecord] = []
        for record in ordered:
            while (
                stack
                and record.start >= stack[-1].end
                and not _contains(stack[-1], record)
            ):
                stack.pop()
            if stack and not _contains(stack[-1], record):
                raise AnalysisError(
                    "partially overlapping IKET ranges on one warp: "
                    f"{stack[-1].name}[{stack[-1].start},{stack[-1].end}] vs "
                    f"{record.name}[{record.start},{record.end}]"
                )
            if stack:
                record.parent = stack[-1].index
                stack[-1].children.append(record.index)
            stack.append(record)


def _ancestors(
    record: RangeRecord, by_index: Mapping[int, RangeRecord]
) -> list[RangeRecord]:
    result: list[RangeRecord] = []
    parent = record.parent
    while parent is not None:
        value = by_index[parent]
        result.append(value)
        parent = value.parent
    return result


def _descendants(
    parent: RangeRecord, by_index: Mapping[int, RangeRecord], name: str | None = None
) -> list[RangeRecord]:
    result: list[RangeRecord] = []
    stack = list(reversed(parent.children))
    while stack:
        index = stack.pop()
        value = by_index[index]
        if name is None or value.name == name:
            result.append(value)
        stack.extend(reversed(value.children))
    return result


def _select_graph_node(
    document: Mapping[str, Any], kernel_pattern: str
) -> tuple[str, dict[str, Any]]:
    regex = re.compile(kernel_pattern)
    graph_launches = document.get("graphLaunches")
    if not isinstance(graph_launches, dict):
        raise AnalysisError("graphLaunches must be an object")
    matches: list[tuple[str, dict[str, Any]]] = []
    for graph_key, nodes in graph_launches.items():
        if not isinstance(nodes, list):
            raise AnalysisError(f"graphLaunches[{graph_key!r}] must be an array")
        for node in nodes:
            if not isinstance(node, dict):
                raise AnalysisError(
                    f"graphLaunches[{graph_key!r}] node is not an object"
                )
            kernel = node.get("kernelName")
            if isinstance(kernel, str) and regex.search(kernel):
                matches.append((str(graph_key), node))
    if len(matches) != 1:
        regular = document.get("launches", [])
        regular_count = (
            sum(
                1
                for node in regular
                if isinstance(node, dict)
                and isinstance(node.get("kernelName"), str)
                and regex.search(node["kernelName"])
            )
            if isinstance(regular, list)
            else 0
        )
        if not matches and regular_count:
            raise AnalysisError(
                f"target exists only as {regular_count} regular/eager launch(es); "
                "a unique CUDA-Graph replay node is required"
            )
        raise AnalysisError(
            f"expected exactly one matching CUDA-Graph node, found {len(matches)}"
        )
    return matches[0]


def _find_single_decoded(root: Path) -> Path:
    candidates = (
        sorted(root.rglob("iket.decoded_results.json")) if root.is_dir() else [root]
    )
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise AnalysisError(
            f"{root}: expected one decoded JSON, found {len(candidates)}"
        )
    return candidates[0]


def _find_pid_manifest(root: Path, pid: int) -> Path:
    candidates: list[Path] = []
    for path in root.rglob("target_manifest.json"):
        if not path.is_file():
            continue
        document = _load_json(path)
        try:
            candidate_pid = _normalize_pid(document.get("pid"), f"{path}: pid")
        except AnalysisError:
            continue
        if candidate_pid == pid:
            candidates.append(path)
    if len(candidates) != 1:
        raise AnalysisError(
            f"{root}: expected one same-PID target_manifest.json for pid={pid}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _overflow_values(
    value: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = (*path, str(key))
            if str(key).lower() in {
                "trace_overflow",
                "context_overflow",
                "buffer_overflow",
            }:
                yield (".".join(next_path), child)
            yield from _overflow_values(child, next_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _overflow_values(child, (*path, str(index)))


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "no", "none", "clear", "not_detected", "0"}:
            return False
        if normalized in {"true", "yes", "detected", "overflow", "1"}:
            return True
    raise AnalysisError(f"invalid trace overflow verdict: {value!r}")


def _manifest_overflow_gate(root: Path, target_manifest: Mapping[str, Any]) -> None:
    """Reject an explicit overflow report; null is left for the raw-data gate."""
    observations = list(_overflow_values(target_manifest))
    for filename in (
        "capture_manifest.json",
        "trace_status.json",
        "iket_capture_manifest.json",
    ):
        for path in root.rglob(filename):
            try:
                if path.stat().st_size <= 16 * 1024 * 1024:
                    observations.extend(_overflow_values(_load_json(path)))
            except OSError as exc:
                raise AnalysisError(
                    f"cannot inspect overflow evidence {path}: {exc}"
                ) from exc
    verdicts = {
        verdict
        for _, value in observations
        for verdict in [_coerce_bool(value)]
        if verdict is not None
    }
    if True in verdicts:
        raise AnalysisError(f"{root}: IKET trace overflow was reported")


def _find_pid_raw(root: Path, pid: int) -> Path:
    candidates = []
    for path in root.rglob("iket.data.json"):
        if path.is_file() and _pid_from_path(path) == pid:
            candidates.append(path)
    if len(candidates) != 1:
        raise AnalysisError(
            f"{root}: expected one same-PID iket.data.json for pid={pid}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _instrument_max_count(root: Path, kernel_name: str) -> tuple[Path, int]:
    candidates = sorted(root.rglob("instrument.config.json"))
    if len(candidates) != 1:
        raise AnalysisError(
            f"{root}: expected one instrument.config.json, found {len(candidates)}"
        )
    path = candidates[0]
    document = _load_json(path)
    configs = document.get("configs")
    if not isinstance(configs, list):
        raise AnalysisError(f"{path}: configs must be an array")
    matches = [
        value
        for value in configs
        if isinstance(value, dict) and value.get("kernel") == kernel_name
    ]
    if len(matches) != 1:
        raise AnalysisError(
            f"{path}: expected one exact-kernel instrumentation config, found {len(matches)}"
        )
    return path, _integer(matches[0].get("maxTsCntPerWarp"), "maxTsCntPerWarp")


def _raw_cta(value: Any) -> tuple[int, int, int]:
    # NativeDump stores a flattened CTA id for a (1,1,Z) grid in 0.7.10.
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, 0, value)
    return _as_xyz(value, "raw warp ctaId")


def _derive_trace_capacity(
    root: Path,
    pid: int,
    *,
    graph_key: str,
    context_id: int,
    grid_id: int,
    kernel_name: str,
    selected_cta: tuple[int, int, int],
) -> dict[str, Any]:
    """Prove headroom from provider 0.7.10 NativeDump buffers.

    ``bytesWritten`` includes the 16-byte buffer header while ``raw_data``
    contains 32-bit words after that header.  NativeDump uses an 8-byte slot
    for this overlay's 32-bit payload events, so total capacity is
    ``16 + maxTsCntPerWarp * 8``.
    """
    raw_path = _find_pid_raw(root, pid)
    raw_document = _load_json(raw_path)
    raw_graph_key, raw_launch = _select_graph_node(raw_document, re.escape(kernel_name))
    if raw_graph_key != graph_key:
        raise AnalysisError("raw/decoded graphLaunchKey drift")
    if _integer(raw_launch.get("contextId"), "raw contextId") != context_id:
        raise AnalysisError("raw/decoded contextId drift")
    if _integer(raw_launch.get("gridId"), "raw gridId") != grid_id:
        raise AnalysisError("raw/decoded gridId drift")
    config_path, config_max = _instrument_max_count(root, kernel_name)
    launch_max = _integer(raw_launch.get("maxTsCntPerWarp"), "raw maxTsCntPerWarp")
    if launch_max != config_max:
        raise AnalysisError(
            f"raw/config maxTsCntPerWarp drift: {launch_max} != {config_max}"
        )
    warps = raw_launch.get("warps")
    if not isinstance(warps, list):
        raise AnalysisError("raw target graph node warps must be an array")
    selected = [
        warp
        for warp in warps
        if isinstance(warp, dict) and _raw_cta(warp.get("ctaId")) == selected_cta
    ]
    warp_ids = [_integer(warp.get("warpId"), "raw warpId") for warp in selected]
    if sorted(warp_ids) != list(range(5)) or len(set(warp_ids)) != 5:
        raise AnalysisError(
            f"raw target CTA must contain exactly warps 0..4, found {sorted(warp_ids)}"
        )
    capacity = 16 + launch_max * 8
    rows = []
    for warp in sorted(
        selected, key=lambda value: _integer(value.get("warpId"), "warpId")
    ):
        warp_id = _integer(warp.get("warpId"), "raw warpId")
        buffer = warp.get("buffer")
        if not isinstance(buffer, dict):
            raise AnalysisError(f"raw warp {warp_id}: buffer must be an object")
        header = buffer.get("header")
        raw_data = buffer.get("raw_data")
        if not isinstance(header, dict) or not isinstance(raw_data, list):
            raise AnalysisError(f"raw warp {warp_id}: missing header/raw_data")
        bytes_written = _integer(header.get("bytesWritten"), "header.bytesWritten")
        expected_bytes = 16 + len(raw_data) * 4
        if bytes_written != expected_bytes:
            raise AnalysisError(
                f"raw warp {warp_id}: bytesWritten={bytes_written}, expected "
                f"16-byte header + raw_data*4 = {expected_bytes}"
            )
        utilization = bytes_written / capacity
        if bytes_written >= capacity:
            raise AnalysisError(
                f"raw warp {warp_id}: buffer reached/exceeded capacity "
                f"({bytes_written}/{capacity})"
            )
        if utilization >= 0.90:
            raise AnalysisError(
                f"raw warp {warp_id}: buffer utilization {utilization:.3%} lacks 10% headroom"
            )
        rows.append(
            {
                "warp_id": warp_id,
                "bytes_written": bytes_written,
                "raw_data_u32_count": len(raw_data),
                "capacity_bytes": capacity,
                "utilization": utilization,
                "headroom_pass": True,
            }
        )
    return {
        "trace_overflow": False,
        "derivation": (
            "provider 0.7.10 NativeDump: bytesWritten == 16 + len(raw_data)*4; "
            "capacity == 16 + maxTsCntPerWarp*8; utilization <90%"
        ),
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "instrument_config": str(config_path),
        "instrument_config_sha256": _sha256(config_path),
        "max_ts_count_per_warp": launch_max,
        "warps": rows,
    }


def _find_tracker_cubin(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.rglob("*.cubin")
        if path.is_file() and "tracker" in path.parts
    )
    if len(candidates) != 1:
        raise AnalysisError(
            f"{root}: expected one tracker cubin, found {len(candidates)}"
        )
    return candidates[0]


def _gate_digest(document: Mapping[str, Any], field: str) -> str | None:
    value = document.get(field)
    if isinstance(value, str):
        return value
    nested_name = field.removesuffix("_sha256")
    nested = document.get(nested_name)
    if isinstance(nested, dict) and isinstance(nested.get("sha256"), str):
        return nested["sha256"]
    artifacts = document.get("artifacts")
    if isinstance(artifacts, dict):
        artifact = artifacts.get(nested_name)
        if isinstance(artifact, dict) and isinstance(artifact.get("sha256"), str):
            return artifact["sha256"]
    identity = document.get("identity")
    if isinstance(identity, dict) and isinstance(identity.get(field), str):
        return identity[field]
    return None


def _load_pc_sass_gate(
    root: Path,
    *,
    tracker_cubin_sha256: str,
    instrument_config_sha256: str,
) -> tuple[frozenset[str], dict[str, Any]]:
    paths = sorted(path for path in root.rglob("pc_sass_gate.json") if path.is_file())
    if not paths:
        return frozenset(), {
            "present": False,
            "pass": False,
            "verified_range_names": [],
            "reason": "pc_sass_gate.json is missing",
        }
    if len(paths) != 1:
        raise AnalysisError(f"{root}: expected at most one pc_sass_gate.json")
    path = paths[0]
    document = _load_json(path)
    verified = document.get("verified_range_names")
    if not isinstance(verified, list) or not all(
        isinstance(item, str) and item for item in verified
    ):
        raise AnalysisError("pc_sass_gate.verified_range_names must be strings")
    gate_tracker_sha = _gate_digest(document, "tracker_cubin_sha256")
    gate_config_sha = _gate_digest(document, "instrument_config_sha256")
    if gate_tracker_sha != tracker_cubin_sha256:
        raise AnalysisError("pc_sass_gate tracker cubin SHA does not match capture")
    if gate_config_sha != instrument_config_sha256:
        raise AnalysisError("pc_sass_gate instrument config SHA does not match capture")
    declared_pass = document.get("overall_pass", document.get("pass"))
    if not isinstance(declared_pass, bool):
        raise AnalysisError("pc_sass_gate.overall_pass must be boolean")
    return frozenset(verified if declared_pass else []), {
        "present": True,
        "pass": declared_pass,
        "path": str(path),
        "gate_sha256": _sha256(path),
        "tracker_cubin_sha256": gate_tracker_sha,
        "instrument_config_sha256": gate_config_sha,
        "verified_range_names": verified,
        "reasons": document.get("reasons", []),
    }


def _task_table(manifest: Mapping[str, Any], label: str) -> list[dict[str, int]]:
    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict):
        raise AnalysisError(f"{label}: workspace must be an object")
    raw = workspace.get("task_table")
    if not isinstance(raw, list):
        raise AnalysisError(f"{label}: workspace.task_table must be an array")
    expected = _integer(
        workspace.get("expected_task_count"), f"{label}: expected_task_count"
    )
    if expected <= 0:
        raise AnalysisError(f"{label}: expected_task_count must be positive")
    if len(raw) != expected:
        raise AnalysisError(
            f"{label}: task table has {len(raw)} rows, expected {expected}"
        )
    result: list[dict[str, int]] = []
    seen: set[int] = set()
    fields = (
        "task_slot",
        "expert",
        "m_tile",
        "slice_begin",
        "slice_count",
        "valid_rows",
        "ready",
    )
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise AnalysisError(f"{label}: task_table[{index}] must be an object")
        parsed = {
            field: _integer(row.get(field), f"task_table[{index}].{field}")
            for field in fields
        }
        slot = parsed["task_slot"]
        if slot in seen:
            raise AnalysisError(f"{label}: duplicate task_slot={slot}")
        seen.add(slot)
        if parsed["slice_count"] <= 0:
            raise AnalysisError(
                f"{label}: task_slot={slot} has non-positive slice_count"
            )
        if parsed["expert"] < 0 or parsed["m_tile"] < 0 or parsed["slice_begin"] < 0:
            raise AnalysisError(f"{label}: task_slot={slot} has a negative descriptor")
        if not 0 < parsed["valid_rows"] <= TILE_M:
            raise AnalysisError(f"{label}: task_slot={slot} has invalid valid_rows")
        if parsed["ready"] != 0:
            raise AnalysisError(
                f"{label}: task_slot={slot} violates deferred ready=0 policy"
            )
        result.append(parsed)
    if seen != set(range(expected)):
        raise AnalysisError(f"{label}: task slots are not exactly [0,{expected})")
    if workspace.get("task_model_pass") is not True:
        raise AnalysisError(f"{label}: task_model_pass is not true")
    tail = _integer(workspace.get("task_tail"), f"{label}: task_tail")
    head = _integer(workspace.get("task_head"), f"{label}: task_head")
    grid_z = _integer(workspace.get("grid_z", 110), f"{label}: grid_z")
    if grid_z != 110 or tail != expected or head != tail + grid_z:
        raise AnalysisError(
            f"{label}: workspace closure failed (tail={tail}, head={head}, "
            f"grid_z={grid_z}, expected={expected})"
        )
    policy = workspace.get("policy")
    if isinstance(policy, dict) and policy.get("full_tile_publish_enabled") != 0:
        raise AnalysisError(f"{label}: full_tile_publish_enabled must be zero")
    row_sum = _integer(workspace.get("row_counts_sum"), f"{label}: row_counts_sum")
    if row_sum != FIXED_ROW_COUNT:
        raise AnalysisError(
            f"{label}: row_counts_sum={row_sum}, expected {FIXED_ROW_COUNT}"
        )
    return sorted(result, key=lambda row: row["task_slot"])


def _manifest_event_evidence(
    manifest: Mapping[str, Any],
) -> dict[str, int | None]:
    evidence = manifest.get("instrumentation_evidence")
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, dict):
        raise AnalysisError("instrumentation_evidence must be an object when present")
    raw_model = evidence.get("qmma_static_count_per_warp_slice")
    if not isinstance(raw_model, dict):
        if raw_model is not None:
            raise AnalysisError(
                "qmma_static_count_per_warp_slice must be an object or null"
            )
        raw_model = {}
    model = dict(DEFAULT_EVENT_MODEL)
    for name in REQUIRED_TASK_PHASES:
        if raw_model.get(name) is not None:
            model[name] = _integer(raw_model[name], f"event model {name}")
    if "fc2_blocks_per_slice" in evidence:
        model["fc2_blocks_per_slice"] = _integer(
            evidence["fc2_blocks_per_slice"], "fc2_blocks_per_slice"
        )
    return model


def load_capture(root: Path, kernel_pattern: str = "MoEDynamicKernel") -> Capture:
    root = root.expanduser().resolve()
    decoded_path = _find_single_decoded(root)
    pid = _pid_from_path(decoded_path)
    manifest_path = _find_pid_manifest(root, pid)
    manifest = _load_json(manifest_path)
    if _normalize_pid(manifest.get("pid"), f"{manifest_path}: pid") != pid:
        raise AnalysisError(f"{manifest_path}: PID does not match decoded trace")
    if manifest.get("status") not in ("pass", "complete", "ok"):
        raise AnalysisError(f"{manifest_path}: target status is not pass/complete/ok")
    _manifest_overflow_gate(root, manifest)
    _task_table(manifest, str(manifest_path))

    capture = manifest.get("capture")
    graph = manifest.get("graph")
    if not isinstance(capture, dict) or not isinstance(graph, dict):
        raise AnalysisError(f"{manifest_path}: capture/graph must be objects")
    capture_id = capture.get("capture_id")
    cluster_id = capture.get("cluster_id")
    if not isinstance(capture_id, str) or not capture_id:
        raise AnalysisError(f"{manifest_path}: capture_id is missing")
    if not isinstance(cluster_id, str) or not cluster_id:
        raise AnalysisError(f"{manifest_path}: cluster_id is missing")
    selected_cta = _as_xyz(capture.get("selected_cta"), "capture.selected_cta")
    manifest_pattern = graph.get("kernel_name_pattern", kernel_pattern)
    if not isinstance(manifest_pattern, str) or not manifest_pattern:
        raise AnalysisError("graph.kernel_name_pattern must be a string")

    document = _load_json(decoded_path)
    graph_key, launch = _select_graph_node(document, manifest_pattern)
    kernel_name = launch.get("kernelName")
    if not isinstance(kernel_name, str) or not kernel_name:
        raise AnalysisError("target graph node is missing kernelName")
    context_id = _integer(launch.get("contextId"), "graph contextId")
    grid_id = _integer(launch.get("gridId"), "graph gridId")
    grid = tuple(
        _integer(launch.get(f"gridDim{axis}"), f"gridDim{axis}") for axis in "XYZ"
    )
    block = tuple(
        _integer(launch.get(f"blockDim{axis}"), f"blockDim{axis}") for axis in "XYZ"
    )
    expected_grid = _as_xyz(graph.get("expected_grid"), "graph.expected_grid")
    expected_block = _as_xyz(graph.get("expected_block"), "graph.expected_block")
    if grid != expected_grid or block != expected_block:
        raise AnalysisError(
            f"dispatch drift: decoded grid/block={grid}/{block}, "
            f"expected={expected_grid}/{expected_block}"
        )
    for identity_field, actual in (
        ("context_id", context_id),
        ("graph_launch_key", graph_key),
        ("grid_id", grid_id),
    ):
        expected = graph.get(identity_field)
        if expected is not None and str(expected) != str(actual):
            raise AnalysisError(
                f"graph identity drift for {identity_field}: {actual!r} != {expected!r}"
            )

    ranges = _parse_ranges(document, launch)
    if not ranges:
        raise AnalysisError("target graph node contains no named ranges")
    user_event_names = _decode_merged_ranges(ranges)
    ctas = {record.location.cta for record in ranges}
    if ctas != {selected_cta}:
        raise AnalysisError(
            f"selected CTA drift: ranges contain {sorted(ctas)}, expected {selected_cta}"
        )
    event_model = _manifest_event_evidence(manifest)
    trace_capacity = _derive_trace_capacity(
        root,
        pid,
        graph_key=graph_key,
        context_id=context_id,
        grid_id=grid_id,
        kernel_name=kernel_name,
        selected_cta=selected_cta,
    )
    tracker_cubin_path = _find_tracker_cubin(root)
    tracker_cubin_sha256 = _sha256(tracker_cubin_path)
    verified, pc_sass_gate = _load_pc_sass_gate(
        root,
        tracker_cubin_sha256=tracker_cubin_sha256,
        instrument_config_sha256=trace_capacity["instrument_config_sha256"],
    )
    return Capture(
        root=root,
        decoded_path=decoded_path,
        manifest_path=manifest_path,
        pid=pid,
        capture_id=capture_id,
        cluster_id=cluster_id,
        selected_cta=selected_cta,
        graph_key=graph_key,
        context_id=context_id,
        grid_id=grid_id,
        kernel_name=kernel_name,
        grid=grid,  # type: ignore[arg-type]
        block=block,  # type: ignore[arg-type]
        user_event_names=user_event_names,
        ranges=ranges,
        manifest=manifest,
        event_model=event_model,
        pc_sass_verified_ranges=verified,
        pc_sass_gate=pc_sass_gate,
        tracker_cubin_path=tracker_cubin_path,
        tracker_cubin_sha256=tracker_cubin_sha256,
        trace_capacity=trace_capacity,
    )


def _stratum(row: Mapping[str, int], task_count: int) -> str:
    slot = row["task_slot"]
    if slot * 10 < task_count:
        position = "early"
    elif slot * 10 >= task_count * 9:
        position = "tail"
    else:
        position = "steady"
    fullness = "full" if row["valid_rows"] == TILE_M else "partial"
    return f"{position}|{fullness}|slices={row['slice_count']}"


def build_population(captures: Sequence[Capture]) -> dict[str, Any]:
    if not captures:
        raise AnalysisError("no captures")
    tables = [
        _task_table(capture.manifest, str(capture.manifest_path))
        for capture in captures
    ]
    canonical = tables[0]
    for capture, table in zip(captures[1:], tables[1:], strict=True):
        if table != canonical:
            raise AnalysisError(
                f"task population identity drift in capture {capture.capture_id}"
            )
    task_count = len(canonical)
    rows = []
    counts: Counter[str] = Counter()
    for row in canonical:
        stratum = _stratum(row, task_count)
        counts[stratum] += 1
        rows.append({**row, "stratum": stratum})
    strata = [
        {
            "stratum": name,
            "population_tasks": count,
            "population_weight": count / task_count,
        }
        for name, count in sorted(counts.items())
    ]
    return {
        "schema_version": 1,
        "task_count": task_count,
        "tasks": rows,
        "strata": strata,
        "source_manifests": [str(capture.manifest_path) for capture in captures],
    }


def _outer_range_gate(capture: Capture) -> None:
    by_warp: dict[WarpLocation, list[RangeRecord]] = defaultdict(list)
    for record in capture.ranges:
        by_warp[record.location].append(record)
    observed_warps = {location.warp for location in by_warp}
    if observed_warps != set(range(5)):
        raise AnalysisError(
            f"capture {capture.capture_id}: decoded target CTA lacks warp tracks 0..4"
        )
    for location, records in by_warp.items():
        outers = [record for record in records if record.name == "moe_kernel"]
        if len(outers) != 1:
            raise AnalysisError(
                f"capture {capture.capture_id} warp={location.warp}: expected one closed "
                f"moe_kernel envelope, found {len(outers)}"
            )
        outer = outers[0]
        if outer.parent is not None:
            raise AnalysisError("moe_kernel envelope is unexpectedly nested")
        if any(not _contains(outer, record) for record in records):
            raise AnalysisError(
                f"capture {capture.capture_id} warp={location.warp}: named range outside "
                "moe_kernel envelope"
            )


def _load_binary_gate(path: Path | None, captures: Sequence[Capture]) -> dict[str, Any]:
    if path is None:
        return {
            "present": False,
            "gate_sha256": None,
            "binary_semantic_omma_gate": {
                "pass": False,
                "control_static_semantic_omma_count": None,
                "candidate_static_semantic_omma_count": None,
                "reason": "--binary-gate was not supplied",
            },
            "formal_dominance": {
                "eligible": False,
                "reasons": ["--binary-gate was not supplied"],
            },
        }
    path = path.expanduser().resolve()
    document = _load_json(path)
    semantic = document.get("binary_semantic_omma_gate")
    formal = document.get("formal_dominance")
    candidate = document.get("candidate")
    if not isinstance(semantic, dict) or not isinstance(formal, dict):
        raise AnalysisError(
            "binary gate requires binary_semantic_omma_gate and formal_dominance objects"
        )
    if not isinstance(candidate, dict):
        raise AnalysisError("binary gate candidate object is missing")
    control_count = _integer(
        semantic.get("control_static_semantic_omma_count"),
        "binary gate control static semantic OMMA count",
    )
    candidate_count = _integer(
        semantic.get("candidate_static_semantic_omma_count"),
        "binary gate candidate static semantic OMMA count",
    )
    semantic_pass = semantic.get("pass")
    if not isinstance(semantic_pass, bool):
        raise AnalysisError("binary_semantic_omma_gate.pass must be boolean")
    if semantic_pass and control_count != candidate_count:
        raise AnalysisError("semantic OMMA gate passes with unequal static counts")
    eligible = formal.get("eligible")
    reasons = formal.get("reasons")
    if (
        not isinstance(eligible, bool)
        or not isinstance(reasons, list)
        or not all(isinstance(reason, str) for reason in reasons)
    ):
        raise AnalysisError(
            "formal_dominance requires eligible bool and reasons strings"
        )
    if not eligible and not reasons:
        raise AnalysisError("ineligible formal_dominance must explain its reasons")
    candidate_sha = candidate.get("cubin_sha256")
    if not isinstance(candidate_sha, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", candidate_sha
    ):
        raise AnalysisError("binary gate candidate cubin SHA-256 is missing")
    tracker_hashes = {capture.tracker_cubin_sha256 for capture in captures}
    if {value.lower() for value in tracker_hashes} != {candidate_sha.lower()}:
        raise AnalysisError(
            "binary gate candidate cubin SHA does not match capture tracker cubin"
        )
    return {
        "present": True,
        "path": str(path),
        "gate_sha256": _sha256(path),
        "candidate_cubin_sha256": candidate_sha.lower(),
        "binary_semantic_omma_gate": {
            "pass": semantic_pass and control_count == candidate_count,
            "control_static_semantic_omma_count": control_count,
            "candidate_static_semantic_omma_count": candidate_count,
            "reason": semantic.get("reason"),
        },
        "formal_dominance": {
            "eligible": eligible,
            "reasons": reasons,
        },
    }


def _resolve_event_model(
    captures: Sequence[Capture], binary_gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve per-slice dynamic counts without source-geometry guesses."""
    resolved: dict[str, int] = {}
    authority: dict[str, str] = {}
    for name in REQUIRED_TASK_PHASES:
        supplied = {
            int(capture.event_model[name])
            for capture in captures
            if capture.event_model.get(name) is not None
        }
        if len(supplied) > 1:
            raise AnalysisError(
                f"same-cubin event model drift for {name}: {sorted(supplied)}"
            )
        observed_counts: list[int] = []
        for capture in captures:
            by_index = {record.index: record for record in capture.ranges}
            for slice_record in (
                record for record in capture.ranges if record.name == "mma_slice"
            ):
                records = _descendants(slice_record, by_index, name)
                if not records:
                    raise AnalysisError(
                        f"capture {capture.capture_id}: closed mma_slice lacks {name}"
                    )
                payloads = [_payload_int(record) for record in records]
                count = len(records)
                if sorted(payloads) != list(range(count)) or len(payloads) != len(
                    set(payloads)
                ):
                    raise AnalysisError(
                        f"capture {capture.capture_id}: {name} payload ordinals are not "
                        f"exactly [0,{count})"
                    )
                observed_counts.append(count)
        if not observed_counts or len(set(observed_counts)) != 1:
            raise AnalysisError(
                f"decoded event count for {name} is missing or inconsistent: "
                f"{sorted(set(observed_counts))}"
            )
        observed = observed_counts[0]
        if supplied and observed != next(iter(supplied)):
            raise AnalysisError(
                f"decoded/same-cubin event count drift for {name}: "
                f"{observed} != {next(iter(supplied))}"
            )
        resolved[name] = observed
        authority[name] = (
            "same-cubin supplied + decoded ordinal closure"
            if supplied
            else "decoded ordinal closure only"
        )
    block_counts = {
        int(capture.event_model["fc2_blocks_per_slice"])
        for capture in captures
        if capture.event_model.get("fc2_blocks_per_slice") is not None
    }
    if len(block_counts) != 1:
        raise AnalysisError("fc2_blocks_per_slice source model is missing or drifting")
    resolved["fc2_blocks_per_slice"] = next(iter(block_counts))
    for capture in captures:
        capture.event_model.update(resolved)

    runtime_event_total = sum(resolved[name] for name in REQUIRED_TASK_PHASES)
    semantic_gate = binary_gate["binary_semantic_omma_gate"]
    binary_gate_pass = bool(semantic_gate["pass"])
    return {
        "resolved_counts_per_warp_slice": resolved,
        "authority": authority,
        "decoded_ordinal_closure": True,
        "runtime_event_total_per_warp_slice": runtime_event_total,
        "candidate_static_semantic_omma_count": semantic_gate[
            "candidate_static_semantic_omma_count"
        ],
        "binary_gate_sha256": binary_gate["gate_sha256"],
        "binary_semantic_omma_gate": semantic_gate,
        "binary_semantic_omma_gate_pass": binary_gate_pass,
        "static_runtime_comparison_policy": (
            "forbidden: static semantic OMMA instruction count and dynamic decoded "
            "range-event count have different denominators"
        ),
        "formal_event_count_closure": binary_gate_pass,
    }


def _calibration(captures: Sequence[Capture]) -> dict[int, dict[str, Any]]:
    samples: dict[int, list[float]] = defaultdict(list)
    for capture in captures:
        for record in capture.ranges:
            if record.name == "marker_calibration":
                samples[record.location.warp].append(record.duration)
    required = set(range(5))
    if set(samples) != required:
        raise AnalysisError(
            f"marker calibration missing warp(s): {sorted(required - set(samples))}"
        )
    return {
        warp: {
            "count": len(values),
            "p95": _percentile(values, 0.95),
            "max": max(values),
        }
        for warp, values in sorted(samples.items())
    }


def _top_level_phase_unions(captures: Sequence[Capture]) -> list[dict[str, Any]]:
    """Report fixed pre-task phases per warp without adding parallel warps.

    These phases are deliberately kept outside the population-weighted task
    estimator: assigning one-time CTA setup to an arbitrary dynamic task slot
    would bias the early/steady/tail strata.  Their per-warp union remains
    visible as separate evidence instead of disappearing from the analysis.
    """
    rows: list[dict[str, Any]] = []
    for capture in captures:
        by_index = {record.index: record for record in capture.ranges}
        by_warp: dict[WarpLocation, list[RangeRecord]] = defaultdict(list)
        for record in capture.ranges:
            by_warp[record.location].append(record)
        for location, records in sorted(by_warp.items(), key=lambda item: item[0]):
            for phase in TOP_LEVEL_PHASES:
                matches = [record for record in records if record.name == phase]
                if len(matches) != 1:
                    raise AnalysisError(
                        f"capture {capture.capture_id} warp={location.warp}: expected one "
                        f"top-level {phase} range, found {len(matches)}"
                    )
                record = matches[0]
                ancestor_names = {
                    ancestor.name for ancestor in _ancestors(record, by_index)
                }
                if "mma_task" in ancestor_names or "tma_task" in ancestor_names:
                    raise AnalysisError(
                        f"capture {capture.capture_id} warp={location.warp}: {phase} "
                        "is unexpectedly nested under a task"
                    )
                intervals = _union([(record.start, record.end)])
                rows.append(
                    {
                        "capture_id": capture.capture_id,
                        "timestamp_unit": TIMESTAMP_UNIT,
                        "pid": capture.pid,
                        "location": location.key,
                        "warp_id": location.warp,
                        "phase": phase,
                        "bucket": "planned",
                        "interval_count": len(intervals),
                        "start": record.start,
                        "end": record.end,
                        "duration": _duration(intervals),
                    }
                )
    return rows


def _pc_sass_gate(
    captures: Sequence[Capture], calibration: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    required: set[str] = set()
    verified_sets = []
    for capture in captures:
        verified_sets.append(set(capture.pc_sass_verified_ranges))
        for record in capture.ranges:
            p95 = calibration.get(record.location.warp, {}).get("p95")
            if (
                record.name in WAIT_LEAVES
                and isinstance(p95, (int, float))
                and record.duration > p95
            ):
                required.add(record.name)
    verified = set.intersection(*verified_sets) if verified_sets else set()
    missing = sorted(required - verified)
    return {
        "capture_gates": [capture.pc_sass_gate for capture in captures],
        "required_ranges": sorted(required),
        "verified_ranges": sorted(verified & required),
        "missing_ranges": missing,
        "pass": not missing,
        "rule": (
            "wait/barrier enters starvation only when duration exceeds same-warp "
            "empty-marker p95 and same-cubin PC/SASS verifies its boundary"
        ),
    }


def _payload_int(record: RangeRecord) -> int:
    if record.payload is None:
        raise AnalysisError(
            f"{record.name} range[{record.index}] is missing start payloadVal"
        )
    return _integer(record.payload, f"{record.name} payloadVal")


def _validate_hierarchy(record: RangeRecord, ancestors: Sequence[RangeRecord]) -> None:
    names = [item.name for item in ancestors]
    if record.name in REQUIRED_TASK_PHASES:
        if "mma_task" not in names or "mma_slice" not in names:
            raise AnalysisError(f"{record.name} is not nested under mma_task/mma_slice")
    if record.name == "fc2_qmma" and "fc2_block" not in names:
        raise AnalysisError("fc2_qmma is not nested under fc2_block")


def _classify_leaf(
    record: RangeRecord,
    calibration_p95: float,
    pc_sass_verified_ranges: frozenset[str],
) -> str:
    if record.name.endswith("_qmma"):
        return "tensor"
    if record.name in PLANNED_LEAVES:
        return "planned"
    if record.name in ORCHESTRATION_LEAVES:
        return "orchestration"
    if record.name in WAIT_LEAVES:
        if record.duration > calibration_p95 and record.name in pc_sass_verified_ranges:
            return "starvation"
        return "unclassified"
    return "unclassified"


def _preceding_orchestration(
    task: RangeRecord, all_on_warp: Sequence[RangeRecord]
) -> list[RangeRecord]:
    previous_task_end = max(
        (
            record.end
            for record in all_on_warp
            if record.name == "mma_task" and record.end <= task.start
        ),
        default=-math.inf,
    )
    return sorted(
        (
            record
            for record in all_on_warp
            if record.name in ORCHESTRATION_LEAVES
            and previous_task_end <= record.start
            and record.end <= task.start
        ),
        key=lambda item: (item.start, item.end, item.index),
    )


def _event_closure(
    task: RangeRecord,
    by_index: Mapping[int, RangeRecord],
    task_row: Mapping[str, int],
    model: Mapping[str, int | None],
) -> dict[str, Any]:
    slices = _descendants(task, by_index, "mma_slice")
    expected_slices = list(
        range(
            task_row["slice_begin"], task_row["slice_begin"] + task_row["slice_count"]
        )
    )
    observed_slices = [_payload_int(record) for record in slices]
    reasons: list[str] = []
    if sorted(observed_slices) != expected_slices or len(observed_slices) != len(
        set(observed_slices)
    ):
        reasons.append(
            f"slice payloads {sorted(observed_slices)} != expected {expected_slices}"
        )
    per_slice: list[dict[str, Any]] = []
    for slice_record in slices:
        slice_payload = _payload_int(slice_record)
        row: dict[str, Any] = {
            "slice": slice_payload,
            "counts": {},
            "payload_contiguous": {},
        }
        for name in REQUIRED_TASK_PHASES:
            records = _descendants(slice_record, by_index, name)
            payloads = [_payload_int(record) for record in records]
            expected_value = model[name]
            if expected_value is None:
                raise AnalysisError(f"event model for {name} is unresolved")
            expected = int(expected_value)
            contiguous = sorted(payloads) == list(range(expected)) and len(
                payloads
            ) == len(set(payloads))
            row["counts"][name] = len(records)
            row["payload_contiguous"][name] = contiguous
            if len(records) != expected or not contiguous:
                reasons.append(
                    f"slice={slice_payload} {name}: count/payload closure "
                    f"{len(records)}/{contiguous}, expected {expected}/true"
                )
        blocks = _descendants(slice_record, by_index, "fc2_block")
        block_payloads = [_payload_int(record) for record in blocks]
        expected_blocks_value = model["fc2_blocks_per_slice"]
        if expected_blocks_value is None:
            raise AnalysisError("fc2_blocks_per_slice is unresolved")
        expected_blocks = int(expected_blocks_value)
        blocks_closed = sorted(block_payloads) == list(range(expected_blocks)) and len(
            block_payloads
        ) == len(set(block_payloads))
        row["fc2_block_count"] = len(blocks)
        row["fc2_block_payload_contiguous"] = blocks_closed
        if len(blocks) != expected_blocks or not blocks_closed:
            reasons.append(
                f"slice={slice_payload} fc2_block closure {len(blocks)}/{blocks_closed}, "
                f"expected {expected_blocks}/true"
            )
        per_slice.append(row)
    return {"pass": not reasons, "reasons": reasons, "slices": per_slice}


def _leaf_context(
    leaf: RangeRecord, by_index: Mapping[int, RangeRecord]
) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "task_slot": None,
        "slice": None,
        "fc2_block": None,
    }
    for parent in _ancestors(leaf, by_index):
        if parent.name == "mma_task":
            result["task_slot"] = _payload_int(parent)
        elif parent.name == "mma_slice":
            result["slice"] = _payload_int(parent)
        elif parent.name == "fc2_block":
            result["fc2_block"] = _payload_int(parent)
    return result


def _qmma_gaps(
    capture: Capture,
    task: RangeRecord,
    leaves: Sequence[dict[str, Any]],
    by_index: Mapping[int, RangeRecord],
) -> list[dict[str, Any]]:
    qmma = [
        record
        for record in _descendants(task, by_index)
        if record.name.endswith("_qmma")
    ]
    grouped: dict[tuple[int, str, int | None], list[RangeRecord]] = defaultdict(list)
    for record in qmma:
        context = _leaf_context(record, by_index)
        if context["slice"] is None:
            raise AnalysisError(f"{record.name} lacks mma_slice ancestor")
        block = context["fc2_block"]
        if record.name == "fc2_qmma" and block is None:
            raise AnalysisError("fc2_qmma lacks fc2_block ancestor")
        grouped[(int(context["slice"]), record.name, block)].append(record)
    bucket_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for leaf in leaves:
        bucket_intervals[leaf["bucket"]].append((leaf["start"], leaf["end"]))
    result: list[dict[str, Any]] = []
    for (slice_payload, phase, fc2_block), records in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1 if item[0][2] is None else item[0][2],
        ),
    ):
        ordered = sorted(records, key=lambda item: (item.start, item.end, item.index))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.start < previous.end:
                raise AnalysisError(f"overlapping QMMA ranges in {phase}")
            start, end = previous.end, current.start
            row: dict[str, Any] = {
                "capture_id": capture.capture_id,
                "timestamp_unit": TIMESTAMP_UNIT,
                "pid": capture.pid,
                "location": task.location.key,
                "warp_id": task.location.warp,
                "task_slot": _payload_int(task),
                "slice": slice_payload,
                "fc2_block": fc2_block,
                "phase": phase,
                "from_ordinal": _payload_int(previous),
                "to_ordinal": _payload_int(current),
                "start": start,
                "end": end,
                "duration": end - start,
            }
            covered = 0.0
            for bucket in BUCKETS[:-1]:
                value = _intersection_duration(bucket_intervals[bucket], start, end)
                row[f"{bucket}_duration"] = value
                covered += value
            row["unclassified_duration"] = max(0.0, (end - start) - covered)
            result.append(row)
    return result


def _percentage(count: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(100.0 * count / denominator, 6)


def _cadence_summary(
    qmma_gaps: Sequence[Mapping[str, Any]],
    calibration: Mapping[str | int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build compact phase distributions from independently measured warp gaps.

    Rows from different warps are pooled only as duration samples.  No gap is
    constructed across warps, and summed sample durations are not elapsed time.
    """

    calibration_p95: dict[int, int | float] = {}
    for raw_warp, evidence in calibration.items():
        if isinstance(raw_warp, bool) or not isinstance(raw_warp, (int, str)):
            raise AnalysisError(f"calibration warp={raw_warp!r} is not an integer")
        try:
            warp = int(raw_warp)
        except ValueError as exc:
            raise AnalysisError(
                f"calibration warp={raw_warp!r} is not an integer"
            ) from exc
        if warp not in MMA_WARPS:
            continue
        if not isinstance(evidence, Mapping):
            raise AnalysisError(f"calibration warp={warp} must be an object")
        p95 = _finite_number(evidence.get("p95"), f"calibration warp={warp} p95")
        if p95 < 0:
            raise AnalysisError(f"calibration warp={warp} p95 must be non-negative")
        calibration_p95[warp] = p95
    if set(calibration_p95) != MMA_WARPS:
        raise AnalysisError(
            "cadence summary requires calibration p95 for MMA warps "
            f"{sorted(MMA_WARPS)}"
        )

    calibration_values = set(calibration_p95.values())
    calibration_report = {
        "mode": "consistent" if len(calibration_values) == 1 else "per_warp",
        "value": next(iter(calibration_values))
        if len(calibration_values) == 1
        else None,
        "per_warp": {
            str(warp): calibration_p95[warp] for warp in sorted(calibration_p95)
        },
    }

    rows_by_phase: dict[str, list[Mapping[str, Any]]] = {
        phase: [] for phase in REQUIRED_TASK_PHASES
    }
    for index, row in enumerate(qmma_gaps):
        phase = row.get("phase")
        if phase not in rows_by_phase:
            raise AnalysisError(f"qmma_gaps[{index}] has unexpected phase {phase!r}")
        rows_by_phase[str(phase)].append(row)

    phases: dict[str, Any] = {}
    for phase in REQUIRED_TASK_PHASES:
        rows = rows_by_phase[phase]
        durations: list[int | float] = []
        gap_count_per_warp = {str(warp): 0 for warp in sorted(MMA_WARPS)}
        bucket_totals: dict[str, int | float] = {bucket: 0 for bucket in BUCKETS}
        exceedance_counts = {1: 0, 2: 0, 4: 0}
        for index, row in enumerate(rows):
            warp = _integer(row.get("warp_id"), f"{phase} gap[{index}] warp_id")
            if warp not in calibration_p95:
                raise AnalysisError(
                    f"{phase} gap[{index}] has uncalibrated warp={warp}"
                )
            duration = _finite_number(
                row.get("duration"), f"{phase} gap[{index}] duration"
            )
            if duration < 0:
                raise AnalysisError(f"{phase} gap[{index}] duration is negative")
            durations.append(duration)
            gap_count_per_warp[str(warp)] += 1
            for bucket in BUCKETS:
                bucket_duration = _finite_number(
                    row.get(f"{bucket}_duration"),
                    f"{phase} gap[{index}] {bucket}_duration",
                )
                if bucket_duration < 0:
                    raise AnalysisError(
                        f"{phase} gap[{index}] {bucket}_duration is negative"
                    )
                bucket_totals[bucket] += bucket_duration
            for multiplier in exceedance_counts:
                if duration > multiplier * calibration_p95[warp]:
                    exceedance_counts[multiplier] += 1

        count = len(durations)
        phases[phase] = {
            "count": count,
            "gap_count_per_warp": gap_count_per_warp,
            "duration": {
                "p50": _percentile(durations, 0.50),
                "p95": _percentile(durations, 0.95),
                "p99": _percentile(durations, 0.99),
                "p99.9": _percentile(durations, 0.999),
                "max": max(durations) if durations else None,
            },
            "exceedance": {
                "gt_calibration_p95": {
                    "count": exceedance_counts[1],
                    "percentage": _percentage(exceedance_counts[1], count),
                },
                "gt_2x_calibration_p95": {
                    "count": exceedance_counts[2],
                    "percentage": _percentage(exceedance_counts[2], count),
                },
                "gt_4x_calibration_p95": {
                    "count": exceedance_counts[4],
                    "percentage": _percentage(exceedance_counts[4], count),
                },
            },
            "total_gap_duration": sum(durations),
            "bucket_duration_totals": bucket_totals,
        }

    return {
        "schema_version": "exp003.cadence_summary.v1",
        "timestamp_unit": TIMESTAMP_UNIT,
        "parallel_warp_warning": (
            "Each gap is formed only between consecutive QMMA ranges on one warp. "
            "Phase distributions pool those gap samples; durations from parallel "
            "warps are never interpreted as elapsed time."
        ),
        "percent_denominator": "gap count within each phase",
        "duration_total_semantics": (
            "raw sum over gap samples; bucket totals use the same raw-duration "
            "denominator and are not production latency"
        ),
        "calibration_p95": calibration_report,
        "phases": phases,
    }


def extract_observations(
    capture: Capture,
    population: Mapping[str, Any],
    calibration: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_index = {record.index: record for record in capture.ranges}
    task_rows = {row["task_slot"]: row for row in population["tasks"]}
    by_warp: dict[WarpLocation, list[RangeRecord]] = defaultdict(list)
    for record in capture.ranges:
        by_warp[record.location].append(record)
        _validate_hierarchy(record, _ancestors(record, by_index))
    observations: list[dict[str, Any]] = []
    leaf_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    seen_task_warps: set[tuple[int, int]] = set()
    for location, records in sorted(by_warp.items(), key=lambda item: item[0]):
        if location.warp not in MMA_WARPS:
            continue
        p95 = calibration[location.warp]["p95"]
        assert isinstance(p95, (int, float))
        for task in sorted(
            (record for record in records if record.name == "mma_task"),
            key=lambda item: (item.start, item.index),
        ):
            slot = _payload_int(task)
            if slot not in task_rows:
                raise AnalysisError(
                    f"capture {capture.capture_id}: task_slot={slot} absent from same-PID table"
                )
            if (slot, location.warp) in seen_task_warps:
                raise AnalysisError(
                    f"capture {capture.capture_id}: duplicate task_slot={slot} on warp={location.warp}"
                )
            seen_task_warps.add((slot, location.warp))
            closure = _event_closure(
                task, by_index, task_rows[slot], capture.event_model
            )
            orchestration = _preceding_orchestration(task, records)
            envelope_start = min(
                (record.start for record in orchestration), default=task.start
            )
            envelope_end = task.end
            if envelope_end <= envelope_start:
                raise AnalysisError("task/warp envelope has non-positive duration")
            descendants = _descendants(task, by_index)
            leaves = [record for record in descendants if not record.children]
            leaves.extend(record for record in orchestration if not record.children)
            categorized: list[dict[str, Any]] = []
            category_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for leaf in sorted(
                leaves, key=lambda item: (item.start, item.end, item.index)
            ):
                bucket = _classify_leaf(
                    leaf, float(p95), capture.pc_sass_verified_ranges
                )
                context = _leaf_context(leaf, by_index)
                row = {
                    "capture_id": capture.capture_id,
                    "timestamp_unit": TIMESTAMP_UNIT,
                    "pid": capture.pid,
                    "location": location.key,
                    "warp_id": location.warp,
                    "task_slot": slot,
                    "stratum": task_rows[slot]["stratum"],
                    "slice": context["slice"],
                    "fc2_block": context["fc2_block"],
                    "name": leaf.name,
                    "raw_name": leaf.raw_name,
                    "payload": leaf.payload,
                    "raw_payload": leaf.raw_payload,
                    "phase": leaf.phase,
                    "start": leaf.start,
                    "end": leaf.end,
                    "duration": leaf.duration,
                    "bucket": bucket,
                    "calibration_p95": p95,
                    "pc_sass_verified": leaf.name in capture.pc_sass_verified_ranges,
                    "ancestor_path": "/".join(
                        item.name for item in reversed(_ancestors(leaf, by_index))
                    ),
                }
                categorized.append(row)
                leaf_rows.append(row)
                if bucket != "unclassified":
                    category_intervals[bucket].append((leaf.start, leaf.end))
            known_intervals = [
                interval
                for bucket in BUCKETS[:-1]
                for interval in category_intervals[bucket]
            ]
            known_union = _duration(known_intervals)
            category_sum = sum(
                _duration(category_intervals[bucket]) for bucket in BUCKETS[:-1]
            )
            if category_sum - known_union > 1e-9:
                raise AnalysisError(
                    f"capture {capture.capture_id} task={slot} warp={location.warp}: "
                    "leaf categories overlap"
                )
            envelope_duration = envelope_end - envelope_start
            if known_union > envelope_duration + 1e-9:
                raise AnalysisError("categorized leaf union exceeds task/warp envelope")
            durations = {
                bucket: _duration(category_intervals[bucket]) for bucket in BUCKETS[:-1]
            }
            durations["unclassified"] = max(0.0, envelope_duration - known_union)
            shares = {
                bucket: durations[bucket] / envelope_duration for bucket in BUCKETS
            }
            observation = {
                "capture_id": capture.capture_id,
                "timestamp_unit": TIMESTAMP_UNIT,
                "cluster_id": capture.cluster_id,
                "pid": capture.pid,
                "location": location.key,
                "warp_id": location.warp,
                "task_slot": slot,
                "stratum": task_rows[slot]["stratum"],
                "envelope_start": envelope_start,
                "envelope_end": envelope_end,
                "envelope_duration": envelope_duration,
                "durations": durations,
                "shares": shares,
                "event_count_closure": closure,
            }
            observations.append(observation)
            gap_rows.extend(_qmma_gaps(capture, task, categorized, by_index))

    # A complete sampled task has all four independent MMA-warp envelopes.
    task_warps: dict[int, set[int]] = defaultdict(set)
    for observation in observations:
        if observation["event_count_closure"]["pass"]:
            task_warps[observation["task_slot"]].add(observation["warp_id"])
    complete_slots = {slot for slot, warps in task_warps.items() if warps == MMA_WARPS}
    for observation in observations:
        observation["complete_task"] = observation["task_slot"] in complete_slots
    return observations, leaf_rows, gap_rows


def _task_level_observations(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for observation in observations:
        if (
            observation.get("complete_task")
            and observation["event_count_closure"]["pass"]
        ):
            capture_id = str(observation["capture_id"])
            grouped[
                (
                    capture_id,
                    str(observation.get("cluster_id", capture_id)),
                    int(observation["task_slot"]),
                    str(observation["stratum"]),
                )
            ].append(observation)
    result = []
    for (capture_id, cluster_id, slot, stratum), rows in sorted(grouped.items()):
        warps = {int(row["warp_id"]) for row in rows}
        if warps != MMA_WARPS or len(rows) != len(MMA_WARPS):
            continue
        # Equal-warp mean: parallel warp durations are never added.
        shares = {
            bucket: statistics.fmean(float(row["shares"][bucket]) for row in rows)
            for bucket in BUCKETS
        }
        result.append(
            {
                "capture_id": capture_id,
                "cluster_id": cluster_id,
                "task_slot": slot,
                "stratum": stratum,
                "warp_count": len(rows),
                "shares": shares,
            }
        )
    return result


def _coverage(
    task_observations: Sequence[Mapping[str, Any]], population: Mapping[str, Any]
) -> dict[str, Any]:
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in task_observations:
        by_stratum[str(observation["stratum"])].append(observation)
    rows = []
    passed = True
    for population_row in population["strata"]:
        name = population_row["stratum"]
        samples = by_stratum.get(name, [])
        tasks = len(samples)
        captures = len({sample["capture_id"] for sample in samples})
        row_pass = tasks >= 8 and captures >= 3
        passed &= row_pass
        rows.append(
            {
                "stratum": name,
                "population_tasks": population_row["population_tasks"],
                "sample_complete_tasks": tasks,
                "sample_capture_count": captures,
                "pass": row_pass,
            }
        )
    return {"pass": passed, "minimum_tasks": 8, "minimum_captures": 3, "strata": rows}


def _weighted_point(
    observations: Sequence[Mapping[str, Any]], population: Mapping[str, Any]
) -> dict[str, float] | None:
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_stratum[str(observation["stratum"])].append(observation)
    if any(row["stratum"] not in by_stratum for row in population["strata"]):
        return None
    point = {bucket: 0.0 for bucket in BUCKETS}
    for row in population["strata"]:
        name = row["stratum"]
        weight = float(row["population_weight"])
        for bucket in BUCKETS:
            point[bucket] += weight * statistics.fmean(
                float(observation["shares"][bucket]) for observation in by_stratum[name]
            )
    return point


def _bootstrap(
    observations: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    clusters = sorted(
        {
            str(observation.get("cluster_id", observation["capture_id"]))
            for observation in observations
        }
    )
    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        cluster_id = str(observation.get("cluster_id", observation["capture_id"]))
        by_cluster[cluster_id].append(observation)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {bucket: [] for bucket in BUCKETS}
    valid = 0
    for _ in range(replicates):
        selected = [rng.choice(clusters) for _ in clusters]
        sample: list[Mapping[str, Any]] = []
        for cluster_id in selected:
            sample.extend(by_cluster[cluster_id])
        point = _weighted_point(sample, population)
        if point is None:
            continue
        valid += 1
        for bucket in BUCKETS:
            draws[bucket].append(point[bucket])
    intervals = {
        bucket: {
            "lower": _percentile(values, 0.025),
            "upper": _percentile(values, 0.975),
        }
        for bucket, values in draws.items()
    }
    return {
        "seed": seed,
        "requested_replicates": replicates,
        "valid_replicates": valid,
        "cluster_unit": "cluster_id (capture_id fallback)",
        "intervals": intervals,
    }


def decide(
    coverage_pass: bool,
    intervals: Mapping[str, Mapping[str, float | None]],
    *,
    minimum_valid_bootstrap: bool = True,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not coverage_pass:
        return "inconclusive", ["coverage gate failed"]
    if not minimum_valid_bootstrap:
        return "inconclusive", ["too few valid capture-cluster bootstrap replicates"]
    if any(
        intervals.get(bucket, {}).get(bound) is None
        for bucket in BUCKETS
        for bound in ("lower", "upper")
    ):
        return "inconclusive", ["one or more bootstrap intervals are unavailable"]
    unclassified_upper = float(intervals["unclassified"]["upper"])  # type: ignore[arg-type]
    if unclassified_upper > 0.20:
        return "inconclusive", ["unclassified upper bound exceeds 20%"]
    for candidate in DECISION_BUCKETS:
        lower = float(intervals[candidate]["lower"])  # type: ignore[arg-type]
        other_uppers = [
            float(intervals[bucket]["upper"])  # type: ignore[arg-type]
            for bucket in BUCKETS
            if bucket != candidate
        ]
        if lower > 0.50 and all(lower > upper for upper in other_uppers):
            return f"{candidate}-dominant", [
                f"{candidate} lower bound >50% and exceeds other mechanism upper bounds"
            ]
    mixed = [
        bucket
        for bucket in DECISION_BUCKETS
        if float(intervals[bucket]["lower"]) >= 0.20  # type: ignore[arg-type]
    ]
    if len(mixed) >= 2:
        return "mixed", [
            f"at least two mechanism lower bounds are >=20%: {', '.join(mixed)}"
        ]
    reasons.append("mechanism intervals do not satisfy dominant or mixed thresholds")
    return "inconclusive", reasons


def aggregate(
    observations: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    *,
    bootstrap_replicates: int = 2000,
    seed: int = BOOTSTRAP_SEED,
    formal_event_count_closure: bool = True,
    formal_dominance_eligible: bool = True,
    formal_dominance_reasons: Sequence[str] = (),
    pc_sass_closure: bool = True,
) -> dict[str, Any]:
    tasks = _task_level_observations(observations)
    coverage = _coverage(tasks, population)
    point = _weighted_point(tasks, population) if coverage["pass"] else None
    if tasks:
        bootstrap = _bootstrap(
            tasks,
            population,
            replicates=bootstrap_replicates,
            seed=seed,
        )
    else:
        bootstrap = {
            "seed": seed,
            "requested_replicates": bootstrap_replicates,
            "valid_replicates": 0,
            "cluster_unit": "cluster_id (capture_id fallback)",
            "intervals": {bucket: {"lower": None, "upper": None} for bucket in BUCKETS},
        }
    minimum_valid = bootstrap["valid_replicates"] >= max(200, bootstrap_replicates // 2)
    if not formal_event_count_closure:
        decision, reasons = (
            "inconclusive",
            [
                "decoded event closure lacks a passing no-marker "
                "control/candidate binary semantic OMMA gate"
            ],
        )
    elif not formal_dominance_eligible:
        decision, reasons = (
            "inconclusive",
            [
                "static binary is not eligible for formal dominance",
                *formal_dominance_reasons,
            ],
        )
    elif not pc_sass_closure:
        decision, reasons = (
            "inconclusive",
            [
                "one or more above-calibration wait/barrier ranges lack same-cubin PC/SASS closure"
            ],
        )
    else:
        decision, reasons = decide(
            coverage["pass"],
            bootstrap["intervals"],
            minimum_valid_bootstrap=minimum_valid,
        )
    return {
        "schema_version": 1,
        "timestamp_unit": TIMESTAMP_UNIT,
        "estimator": (
            "equal-warp mean within complete tasks; stratum mean; "
            "population task-count weighting"
        ),
        "parallel_warp_rule": "warp intervals are never summed",
        "formal_event_count_closure": formal_event_count_closure,
        "formal_dominance_eligible": formal_dominance_eligible,
        "formal_dominance_reasons": list(formal_dominance_reasons),
        "pc_sass_closure": pc_sass_closure,
        "coverage": coverage,
        "complete_task_observations": tasks,
        "population_weighted_point": point,
        "bootstrap": bootstrap,
        "decision": decision,
        "decision_reasons": reasons,
    }


def _identity_drift_gate(captures: Sequence[Capture]) -> None:
    if len({capture.capture_id for capture in captures}) != len(captures):
        raise AnalysisError("capture_id is not unique")
    fixed = {
        (
            capture.kernel_name,
            capture.grid,
            capture.block,
            tuple(sorted(capture.event_model.items())),
            capture.pc_sass_verified_ranges,
            capture.tracker_cubin_sha256,
            capture.trace_capacity["instrument_config_sha256"],
        )
        for capture in captures
    }
    if len(fixed) != 1:
        raise AnalysisError(
            "kernel/dispatch/instrumentation evidence drift across captures"
        )
    fingerprints = set()
    runtime_fingerprints = set()
    for capture in captures:
        identity = capture.manifest.get("identity")
        if not isinstance(identity, dict):
            raise AnalysisError("identity manifest must be an object")
        before = identity.get("jit_before")
        after = identity.get("jit_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise AnalysisError("target identity lacks jit_before/jit_after")
        before_hash = before.get("jit_artifact_set_sha256")
        after_hash = after.get("jit_artifact_set_sha256")
        if (
            not isinstance(before_hash, str)
            or not before_hash
            or before_hash != after_hash
        ):
            raise AnalysisError(
                f"capture {capture.capture_id}: instrumented JIT artifact set drift"
            )
        if identity.get("jit_artifact_set_stable", True) is not True:
            raise AnalysisError(
                f"capture {capture.capture_id}: JIT stability gate failed"
            )
        candidate_hash = identity.get("candidate_manifest_sha256")
        if not isinstance(candidate_hash, str) or not candidate_hash:
            raise AnalysisError("candidate manifest identity is missing")
        fingerprints.add((candidate_hash, before_hash))

        runtime = capture.manifest.get("runtime")
        if runtime is not None:
            if not isinstance(runtime, dict):
                raise AnalysisError("runtime identity must be an object")
            stable_runtime = {
                key: value
                for key, value in runtime.items()
                # Ephemeral Docker assigns a new container hostname to each
                # capture.  The physical GPU UUID, source, provider, image and
                # toolchain remain in the retained runtime fields and are the
                # durable execution identity.
                if key not in {"timestamp_unix", "pid", "hostname"}
            }
            runtime_fingerprints.add(
                json.dumps(stable_runtime, sort_keys=True, separators=(",", ":"))
            )
    if len(fingerprints) != 1:
        raise AnalysisError("candidate/JIT identity drift across captures")
    if len(runtime_fingerprints) > 1:
        raise AnalysisError("runtime/source/provider identity drift across captures")


def analyze_experiment(
    roots: Sequence[Path],
    *,
    kernel_pattern: str = "MoEDynamicKernel",
    bootstrap_replicates: int = 2000,
    seed: int = BOOTSTRAP_SEED,
    binary_gate_path: Path | None = None,
) -> dict[str, Any]:
    captures = [load_capture(root, kernel_pattern) for root in roots]
    if not captures:
        raise AnalysisError("no capture roots supplied")
    _identity_drift_gate(captures)
    for capture in captures:
        _outer_range_gate(capture)
    binary_gate_evidence = _load_binary_gate(binary_gate_path, captures)
    event_count_evidence = _resolve_event_model(captures, binary_gate_evidence)
    population = build_population(captures)
    calibration = _calibration(captures)
    top_level_phase_unions = _top_level_phase_unions(captures)
    pc_sass_evidence = _pc_sass_gate(captures, calibration)
    observations: list[dict[str, Any]] = []
    leaves: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for capture in captures:
        capture_observations, capture_leaves, capture_gaps = extract_observations(
            capture, population, calibration
        )
        observations.extend(capture_observations)
        leaves.extend(capture_leaves)
        gaps.extend(capture_gaps)
    weighted = aggregate(
        observations,
        population,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        formal_event_count_closure=bool(
            event_count_evidence["formal_event_count_closure"]
        ),
        formal_dominance_eligible=bool(
            binary_gate_evidence["formal_dominance"]["eligible"]
        ),
        formal_dominance_reasons=binary_gate_evidence["formal_dominance"]["reasons"],
        pc_sass_closure=bool(pc_sass_evidence["pass"]),
    )
    return {
        "schema_version": 1,
        "timestamp_unit": TIMESTAMP_UNIT,
        "captures": [capture.identity for capture in captures],
        "calibration": {str(key): value for key, value in calibration.items()},
        "top_level_phase_unions": top_level_phase_unions,
        "event_count_evidence": event_count_evidence,
        "binary_gate_evidence": binary_gate_evidence,
        "pc_sass_evidence": pc_sass_evidence,
        "task_population": population,
        "observations": observations,
        "leaf_intervals": leaves,
        "qmma_gaps": gaps,
        "weighted_phase_shares": weighted,
    }


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float | None) -> str:
    return "missing" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(result: Mapping[str, Any]) -> str:
    weighted = result["weighted_phase_shares"]
    lines = [
        "# Exp 003 IKET cadence breakdown",
        "",
        f"- Evidence scope: {len(result['captures'])} selected-CTA captures; timestamps are `{TIMESTAMP_UNIT}`.",
        "- Parallel MMA-warp intervals were analyzed independently and were never added.",
        f"- Coverage gate: **{'pass' if weighted['coverage']['pass'] else 'fail'}**.",
        f"- Decision: **{weighted['decision']}**.",
        "- Trace-capacity gate: **pass** for every target warp; NativeDump uses a 16-byte header, "
        "`bytesWritten == 16 + len(raw_data)*4`, capacity `16 + maxTsCntPerWarp*8`, and <90% utilization.",
        (
            "- Decoded-event and binary semantic OMMA gates: **pass**."
            if result["event_count_evidence"]["formal_event_count_closure"]
            else "- Binary semantic OMMA gate: **missing or failed**; decoded diagnostics remain available, but the decision is forced inconclusive."
        ),
        (
            "- Static binary formal-dominance eligibility: **pass**."
            if weighted["formal_dominance_eligible"]
            else "- Static binary formal-dominance eligibility: **missing or failed**; diagnostics remain available."
        ),
        (
            "- Above-calibration wait/barrier PC/SASS closure: **pass**."
            if result["pc_sass_evidence"]["pass"]
            else "- Above-calibration wait/barrier PC/SASS closure: **missing**; missing evidence is not zero starvation."
        ),
        "",
        "| Bucket | Weighted point | Bootstrap 95% interval |",
        "|---|---:|---:|",
    ]
    point = weighted["population_weighted_point"]
    intervals = weighted["bootstrap"]["intervals"]
    for bucket in BUCKETS:
        bucket_point = None if point is None else point[bucket]
        interval = intervals[bucket]
        lines.append(
            f"| {bucket} | {_pct(bucket_point)} | "
            f"{_pct(interval['lower'])} – {_pct(interval['upper'])} |"
        )
    lines.extend(["", "Coverage is defined per population stratum:", ""])
    lines.extend(
        f"- `{row['stratum']}`: {row['sample_complete_tasks']} complete tasks / "
        f"{row['sample_capture_count']} captures — {'pass' if row['pass'] else 'fail'}"
        for row in weighted["coverage"]["strata"]
    )
    lines.extend(
        [
            "",
            "The estimate is an instrumented sampled-warp diagnostic. It is not production latency and is not an NCU active-cycle denominator.",
            "Top-level fixed phases are reported per warp separately and are not assigned to an arbitrary dynamic task stratum.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(result: Mapping[str, Any], output_dir: Path) -> None:
    cadence_summary = _cadence_summary(result["qmma_gaps"], result["calibration"])
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise AnalysisError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "exp003_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "task_population.json").write_text(
        json.dumps(result["task_population"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "weighted_phase_shares.json").write_text(
        json.dumps(result["weighted_phase_shares"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cadence_summary.json").write_text(
        json.dumps(cadence_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(render_markdown(result), encoding="utf-8")
    _write_csv(
        output_dir / "top_level_phase_unions.csv",
        result["top_level_phase_unions"],
        (
            "capture_id",
            "timestamp_unit",
            "pid",
            "location",
            "warp_id",
            "phase",
            "bucket",
            "interval_count",
            "start",
            "end",
            "duration",
        ),
    )
    _write_csv(
        output_dir / "warp_leaf_intervals.csv",
        result["leaf_intervals"],
        (
            "capture_id",
            "timestamp_unit",
            "pid",
            "location",
            "warp_id",
            "task_slot",
            "stratum",
            "slice",
            "fc2_block",
            "name",
            "raw_name",
            "payload",
            "raw_payload",
            "phase",
            "start",
            "end",
            "duration",
            "bucket",
            "calibration_p95",
            "pc_sass_verified",
            "ancestor_path",
        ),
    )
    _write_csv(
        output_dir / "qmma_gaps.csv",
        result["qmma_gaps"],
        (
            "capture_id",
            "timestamp_unit",
            "pid",
            "location",
            "warp_id",
            "task_slot",
            "slice",
            "fc2_block",
            "phase",
            "from_ordinal",
            "to_ordinal",
            "start",
            "end",
            "duration",
            "tensor_duration",
            "planned_duration",
            "starvation_duration",
            "orchestration_duration",
            "unclassified_duration",
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--kernel", default="MoEDynamicKernel")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--binary-gate", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.bootstrap_replicates < 1:
            raise AnalysisError("--bootstrap-replicates must be positive")
        result = analyze_experiment(
            args.captures,
            kernel_pattern=args.kernel,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            binary_gate_path=args.binary_gate,
        )
        write_outputs(result, args.output_dir)
        print(args.output_dir.expanduser().resolve(strict=False))
        return 0
    except (AnalysisError, OSError, re.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
