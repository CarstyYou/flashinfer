#!/usr/bin/env python3
"""Thin per-process correctness and E2E worker for exp_016.

This worker deliberately reuses the locked exp_014 runtime/identity protocol
and the exp_005 fixture/oracle/workspace helpers.  It adds only the exp_016
contracts: an always-[E] FC1 input scale, the candidate's token-unit producer
counter, and physical-row-independent hashes of the Route/Q0 payload.

``validate-case`` runs one correctness case and emits compact JSON only.
``measure`` runs one position of a caller-orchestrated A-B-B-A group.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parent
EXP014 = ROOT.parent / "exp_014_scatter_8warp"
EXP005 = ROOT.parent / "exp_005_8warp_spill_reduction"
for dependency in (EXP014, EXP005):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import run_exp014_arm as reused  # noqa: E402


worker = reused.worker
common = reused.common

BASELINE = "baseline_pair_major"
CANDIDATE = "candidate_token_major_reuse"
ARMS = (BASELINE, CANDIDATE)
ABBA = (BASELINE, CANDIDATE, CANDIDATE, BASELINE)
EXPECTED_OVERLAY_SHA256 = {
    BASELINE: "c88cef63492b60c0a77484b50f6400b83a103d168e1535b78972341503810184",
    CANDIDATE: "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19",
}
SCALE_KINDS = ("equal", "unequal")
FIXTURES = ("canonical", "hot_expert", "tail_129")

EXPECTED_GRID = (1, 1, 110)
EXPECTED_BLOCK = (288, 1, 1)
NUM_CTA_WARPS = EXPECTED_BLOCK[0] // 32
TILE_M = 128
WARMUP = 5
ITERS = 50
L2_FLUSH_BYTES = 192 << 20
UNEQUAL_SCALE_PATTERN = (0.5, 1.0, 2.0, 1.0)


def producer_counter_contract(
    arm: str,
    m: int,
    *,
    topk: int = common.TOPK,
    num_cta_warps: int = NUM_CTA_WARPS,
    grid_z: int = EXPECTED_GRID[2],
) -> dict[str, int | str]:
    """Return the arm-specific terminal producer-counter contract.

    Every resident CTA performs one final unsuccessful claim.  Baseline stores
    routed-pair units in ``pair_head``; Candidate retains the ABI field name but
    stores token units.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if m <= 0 or topk <= 0 or num_cta_warps <= 0 or grid_z <= 0:
        raise ValueError("producer-counter geometry must be positive")
    if arm == BASELINE:
        unit = "routed_pair"
        work = m * topk
        claim = num_cta_warps * 2
    else:
        unit = "token"
        work = m
        claim = num_cta_warps
    productive_claims = math.ceil(work / claim)
    terminal = (productive_claims + grid_z) * claim
    return {
        "unit": unit,
        "work": work,
        "claim": claim,
        "productive_claims": productive_claims,
        "terminal": terminal,
    }


def make_input_scale(
    num_experts: int,
    scale_kind: str,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the locked [E] scale tensor without scalar normalization."""
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if scale_kind == "equal":
        return torch.ones(num_experts, dtype=torch.float32, device=device)
    if scale_kind != "unequal":
        raise ValueError(f"unknown scale kind: {scale_kind}")
    pattern = torch.tensor(UNEQUAL_SCALE_PATTERN, dtype=torch.float32, device=device)
    repeats = math.ceil(num_experts / pattern.numel())
    return pattern.repeat(repeats)[:num_experts].contiguous()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def apply_input_scale(fixture_module: Any, weights: Any, scale_kind: str) -> Any:
    scale = make_input_scale(
        common.E, scale_kind, device=weights.w1_global_scale.device
    )
    if tuple(scale.shape) != (common.E,):
        raise RuntimeError("exp_016 input scale must remain [E]")
    unique = sorted(float(value) for value in torch.unique(scale).cpu().tolist())
    if scale_kind == "equal" and unique != [1.0]:
        raise RuntimeError("equal scale fixture drift")
    if scale_kind == "unequal" and unique != [0.5, 1.0, 2.0]:
        raise RuntimeError("unequal scale fixture drift")
    manifest = dict(weights.manifest)
    manifest.update(
        {
            "exp016_input_scale_kind": scale_kind,
            "w1_global_scale_shape": list(scale.shape),
            "w1_global_scale_values": unique,
            "w1_global_scale_sha256": fixture_module.tensor_sha256(scale),
            "share_input_across_experts_expected": False,
        }
    )
    return replace(weights, w1_global_scale=scale, manifest=manifest)


def reference_weights_for_input_scale(weights: Any, scale_kind: str) -> Any:
    """Return weights for the independent mathematical oracle.

    In this fused kernel ``w1_alpha`` has two coupled roles: P3 quantizes X
    with the per-expert value and the FC1 epilogue multiplies the accumulator
    by the same value.  Those factors cancel; changing only the runtime input
    scale must not change the represented W1 matrix.  The repository fixture's
    generic dequantizer instead interprets ``w1_global_scale`` as a weight-only
    divisor, which is valid for its canonical all-ones fixture but is not a
    reference for this directed unequal-scale test.  Keep the packed weights
    and block scales unchanged and use ones for the oracle's weight divisor.
    """
    if scale_kind == "equal":
        return weights
    if scale_kind != "unequal":
        raise ValueError(f"unknown scale kind: {scale_kind}")
    oracle_scale = torch.ones_like(weights.w1_global_scale)
    manifest = dict(weights.manifest)
    manifest.update(
        {
            "exp016_oracle_w1_scale_semantics": (
                "runtime input-scale and FC1 epilogue factors cancel; packed "
                "W1 representation remains canonical"
            ),
            "exp016_oracle_w1_global_scale_sha256": tensor_sha256(oracle_scale),
        }
    )
    return replace(weights, w1_global_scale=oracle_scale, manifest=manifest)


def scale_storage_offsets(
    physical_rows: torch.Tensor,
    sf_blocks: int,
    *,
    tile_m: int = TILE_M,
) -> torch.Tensor:
    """Map logical per-row SF indices to the kernel's swizzled flat storage."""
    if physical_rows.ndim != 1:
        raise ValueError("physical_rows must be one-dimensional")
    if sf_blocks <= 0 or tile_m != 128:
        raise ValueError("expected positive SF blocks and the locked M128 tile")
    rows = physical_rows.to(dtype=torch.int64)
    sf = torch.arange(sf_blocks, dtype=torch.int64, device=rows.device)
    physical_tile = torch.div(rows, tile_m, rounding_mode="floor")[:, None]
    tile_row = (rows % tile_m)[:, None]
    num_k_tiles = math.ceil(sf_blocks / 4)
    tile_stride = num_k_tiles * 32 * 4 * 4
    return (
        physical_tile * tile_stride
        + torch.div(sf, 4, rounding_mode="floor")[None, :] * (32 * 4 * 4)
        + (tile_row % 32) * (4 * 4)
        + torch.div(tile_row % 128, 32, rounding_mode="floor") * 4
        + (sf % 4)[None, :]
    )


def _float32_bits(value: torch.Tensor) -> list[int]:
    cpu = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return [int(item) & 0xFFFFFFFF for item in cpu.view(torch.int32).tolist()]


def canonical_route_plan(
    *,
    row_counts: torch.Tensor,
    expert_tile_base: torch.Tensor,
    token_map: torch.Tensor,
    token_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tile_m: int = TILE_M,
) -> dict[str, Any]:
    """Canonicalize physical rows as (expert, token, occurrence, weight bits)."""
    counts = row_counts.detach().to(dtype=torch.int64, device="cpu").tolist()
    bases = expert_tile_base.detach().to(dtype=torch.int64, device="cpu").tolist()
    ids = topk_ids.detach().to(dtype=torch.int64, device="cpu").contiguous()
    weights = topk_weights.detach().to(dtype=torch.float32, device="cpu").contiguous()
    observed_tokens = token_map.detach().to(dtype=torch.int64, device="cpu")
    observed_weight_bits = _float32_bits(token_weights)
    expected_weight_bits = _float32_bits(weights.reshape(-1))

    if ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("topk ids/weights must have the same two-dimensional shape")
    num_experts = len(counts)
    if len(bases) != num_experts + 1:
        raise ValueError("expert_tile_base must contain E+1 entries")

    expected: dict[tuple[int, int], tuple[int, int]] = {}
    expected_counts = [0] * num_experts
    topk = ids.shape[1]
    for token in range(ids.shape[0]):
        for occurrence in range(topk):
            expert = int(ids[token, occurrence])
            if not 0 <= expert < num_experts:
                raise ValueError(f"expert id out of range: {expert}")
            key = (expert, token)
            if key in expected:
                raise ValueError("duplicate expert for one token is outside exp_016")
            expected[key] = (
                occurrence,
                expected_weight_bits[token * topk + occurrence],
            )
            expected_counts[expert] += 1
    if counts != expected_counts:
        raise ValueError("observed row_counts do not match routed experts")

    records: list[tuple[tuple[int, int, int, int], int]] = []
    seen: set[tuple[int, int]] = set()
    for expert, count in enumerate(counts):
        expected_tiles = math.ceil(count / tile_m) if count else 0
        if bases[expert + 1] - bases[expert] != expected_tiles:
            raise ValueError("expert_tile_base does not match row_counts")
        physical_base = bases[expert] * tile_m
        for local_row in range(count):
            physical_row = physical_base + local_row
            token = int(observed_tokens[physical_row])
            route_key = (expert, token)
            if route_key not in expected:
                raise ValueError(f"unexpected physical route {route_key}")
            if route_key in seen:
                raise ValueError(f"duplicate physical route {route_key}")
            seen.add(route_key)
            occurrence, expected_bits = expected[route_key]
            observed_bits = observed_weight_bits[physical_row]
            if observed_bits != expected_bits:
                raise ValueError(f"route weight mismatch for {route_key}")
            records.append(((expert, token, occurrence, observed_bits), physical_row))
    if seen != set(expected):
        raise ValueError("physical rows contain a route hole")

    records.sort(key=lambda item: item[0])
    keys = torch.tensor([item[0] for item in records], dtype=torch.int64)
    physical_rows = torch.tensor([item[1] for item in records], dtype=torch.int64)
    return {
        "keys": keys,
        "physical_rows": physical_rows,
        "logical_routes": len(records),
        "duplicate_routes": 0,
        "missing_routes": 0,
    }


def canonical_logical_payload_digest(
    workspace: Any,
    fixture: Any,
    *,
    hidden_size: int = common.H,
    tile_m: int = TILE_M,
) -> dict[str, Any]:
    """Hash Route/Q0 payload in logical-route order without retaining tensors."""
    plan = canonical_route_plan(
        row_counts=workspace.row_counts,
        expert_tile_base=workspace.expert_tile_base,
        token_map=workspace.token_map,
        token_weights=workspace.token_weights,
        topk_ids=fixture.topk_ids,
        topk_weights=fixture.topk_weights,
        tile_m=tile_m,
    )
    physical_rows = plan["physical_rows"]
    device_rows = physical_rows.to(workspace.packed_input.device)
    packed_cols = hidden_size // 2
    sf_blocks = hidden_size // 16
    packed_matrix = workspace.packed_input.reshape(-1, packed_cols)
    if packed_matrix.shape[0] <= int(physical_rows.max().item()):
        raise ValueError("packed input is shorter than the observed physical rows")
    packed = packed_matrix.index_select(0, device_rows).detach().cpu()

    offsets = scale_storage_offsets(device_rows, sf_blocks, tile_m=tile_m)
    scale_flat = workspace.packed_input_scale.reshape(-1)
    if int(offsets.max().item()) >= scale_flat.numel():
        raise ValueError("scale storage offset exceeds the workspace")
    scales = scale_flat[offsets].detach().cpu()
    component_hashes = {
        "route_metadata_sha256": tensor_sha256(plan["keys"]),
        "packed_fp4_sha256": tensor_sha256(packed),
        "sfa_sha256": tensor_sha256(scales),
    }
    return {
        "schema": "exp016.canonical-route-q0-payload.v1",
        "canonical_key": "(expert,token,occurrence,weight_f32_bits)",
        "logical_routes": plan["logical_routes"],
        "packed_shape": list(packed.shape),
        "sfa_shape": list(scales.shape),
        **component_hashes,
        "combined_sha256": common.canonical_sha256(component_hashes),
        "gate_pass": (
            plan["logical_routes"] == fixture.m * fixture.topk_ids.shape[1]
            and plan["duplicate_routes"] == 0
            and plan["missing_routes"] == 0
        ),
        "invalid_padding_boundary": (
            "only valid logical rows are hashed; padding is not required to be "
            "initialized because task_valid_rows excludes it"
        ),
    }


def corrected_workspace_snapshot(
    arm: str, wrapper: Any, fixture: Any
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors, summary = worker._workspace_snapshot(
        wrapper, fixture, num_cta_warps=NUM_CTA_WARPS
    )
    verification = dict(summary["verification"])
    checks = dict(verification["checks"])
    contract = producer_counter_contract(arm, fixture.m)
    observed = int(wrapper._dynamic_workspace.pair_head.item())
    checks["pair_head_terminal_state"] = observed == contract["terminal"]
    verification.update(
        {
            "checks": checks,
            "gate_pass": all(checks.values()),
            "producer_counter_unit": contract["unit"],
            "producer_claim_count": contract["claim"],
            "productive_claims": contract["productive_claims"],
            "expected_pair_head": contract["terminal"],
            "observed_pair_head": observed,
        }
    )
    summary["verification"] = verification
    summary["exp016_producer_counter_contract"] = contract
    return tensors, summary


def specialization_contract(overlay: Path) -> dict[str, Any]:
    dispatch = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
    )
    keys = [key for key in dispatch._DYNAMIC_KERNEL_CACHE if key[0] == "dynamic"]
    if not keys:
        raise RuntimeError("dynamic kernel cache is empty after launch")
    records = [
        {
            "input_scales_are_reciprocal": bool(key[9]),
            "fast_math": bool(key[10]),
            "share_input_across_experts": bool(key[-1]),
        }
        for key in keys
    ]
    dynamic_gate = all(
        not record["input_scales_are_reciprocal"]
        and record["fast_math"]
        and not record["share_input_across_experts"]
        for record in records
    )
    source = overlay.read_text(encoding="utf-8")
    zero_assignments = re.findall(
        r"^\s*full_tile_publish_enabled\s*=\s*Int32\(0\)\s*$",
        source,
        flags=re.MULTILINE,
    )
    all_assignments = re.findall(
        r"^\s*full_tile_publish_enabled\s*=.*$", source, flags=re.MULTILINE
    )
    publish_gate = len(zero_assignments) == 1 and len(all_assignments) == 1
    payload = {
        "dynamic_cache_entries": len(keys),
        "dynamic_records": records,
        "full_tile_publish_zero_assignment_count": len(zero_assignments),
        "full_tile_publish_all_assignment_count": len(all_assignments),
        "gate_pass": dynamic_gate and publish_gate,
    }
    if not payload["gate_pass"]:
        raise RuntimeError(f"exp_016 specialization contract failed: {payload}")
    return payload


def make_case(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    fixture_module = worker.load_fixture_module()
    device = torch.device("cuda", args.device_index)
    base = fixture_module.make_routed_fixture(args.m, device=device, seed=args.seed)
    if args.fixture == "canonical":
        fixture = base
    else:
        fixture = worker.make_directed_fixture(fixture_module, base, args.fixture)
    weights = fixture_module.make_canonical_weights(device=device, seed=args.seed)
    weights = apply_input_scale(fixture_module, weights, args.scale_kind)
    return fixture_module, fixture, weights


def _artifacts(jit_root: Path) -> tuple[list[dict[str, Any]], str, list[str]]:
    artifacts = common.artifact_manifest(jit_root)
    artifact_hash = common.canonical_sha256(artifacts)
    cubins = sorted(
        {str(item["sha256"]) for item in artifacts if item["path"].endswith(".cubin")}
    )
    return artifacts, artifact_hash, cubins


def _validate_registered_jit(args: argparse.Namespace) -> None:
    if args.jit_policy == "fresh":
        common.require_empty_directory(args.jit_root)
        return
    if not args.expected_jit_artifact_set_sha256:
        raise RuntimeError("reuse requires --expected-jit-artifact-set-sha256")
    _, observed, _ = _artifacts(args.jit_root)
    if observed != args.expected_jit_artifact_set_sha256:
        raise RuntimeError(
            f"registered JIT artifact drift: {observed} != "
            f"{args.expected_jit_artifact_set_sha256}"
        )


def validate_case(args: argparse.Namespace, runtime: Mapping[str, Any]) -> int:
    fixture_module, fixture, weights = make_case(args)
    oracle_weights = reference_weights_for_input_scale(weights, args.scale_kind)
    reference = fixture_module.reference_moe_nvfp4(fixture, oracle_weights)
    arm = worker.build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    specialization = specialization_contract(args.overlay)

    output_values = []
    replay_records = []
    for replay in range(args.replays):
        output, elapsed_ms = arm.replay(sentinel=True)
        output = output.clone()
        _, workspace = corrected_workspace_snapshot(args.arm, arm.wrapper, fixture)
        logical_payload = canonical_logical_payload_digest(
            arm.wrapper._dynamic_workspace, fixture
        )
        diagnostics = fixture_module.output_diagnostics(output, reference)
        reference_error = worker.tensor_error(output, reference)
        nan_remaining = int(torch.isnan(output).sum().item())
        replay_records.append(
            {
                "replay": replay,
                "event_elapsed_us": elapsed_ms * 1000.0,
                "output_sha256": worker.tensor_sha256(output),
                "reference_error": reference_error,
                "oracle": diagnostics,
                "sentinel_nan_remaining": nan_remaining,
                "workspace": workspace,
                "logical_payload": logical_payload,
                "gate_pass": (
                    bool(diagnostics["formal_pass"])
                    and bool(diagnostics["finite"])
                    and nan_remaining == 0
                    and bool(workspace["verification"]["gate_pass"])
                    and bool(logical_payload["gate_pass"])
                ),
            }
        )
        output_values.append(output.detach().cpu())

    stability = worker.tensor_error(output_values[-1], output_values[0])
    stability_gate = reused.self_drift_gate(stability)
    payload_hashes = {
        record["logical_payload"]["combined_sha256"] for record in replay_records
    }
    payload_stable = len(payload_hashes) == 1
    artifacts, artifact_hash, cubins = _artifacts(args.jit_root)
    if not cubins:
        raise RuntimeError("validation produced no retained cubin")
    if (
        args.jit_policy == "reuse"
        and artifact_hash != args.expected_jit_artifact_set_sha256
    ):
        raise RuntimeError("validation mutated the registered JIT artifact set")

    gate_pass = (
        all(bool(record["gate_pass"]) for record in replay_records)
        and bool(stability_gate["gate_pass"])
        and payload_stable
        and bool(specialization["gate_pass"])
    )
    payload = {
        "schema": "exp016.validation-case.v1",
        "status": "complete" if gate_pass else "failed",
        "arm": args.arm,
        "m": args.m,
        "fixture": args.fixture,
        "scale_kind": args.scale_kind,
        "case_identity": {
            "fixture": fixture.manifest,
            "weights": weights.manifest,
            "oracle_weights": oracle_weights.manifest,
            "reference_sha256": worker.tensor_sha256(reference),
        },
        "runtime": dict(runtime),
        "specialization": specialization,
        "producer_counter": producer_counter_contract(args.arm, args.m),
        "replays": replay_records,
        "output_stability": stability,
        "output_stability_gate": stability_gate,
        "logical_payload_replay_stable": payload_stable,
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": artifact_hash,
        "cubin_sha256": cubins,
        "gate_pass": gate_pass,
    }
    common.write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if not gate_pass:
        raise RuntimeError("exp_016 validation case failed")
    return 0


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure(args: argparse.Namespace, runtime: Mapping[str, Any]) -> int:
    if ABBA[args.position] != args.arm:
        raise RuntimeError(
            f"ABBA position {args.position} requires {ABBA[args.position]}"
        )
    _, fixture, weights = make_case(args)
    arm = worker.build_arm(args, fixture, weights)
    arm.eager()
    arm.capture()
    specialization = specialization_contract(args.overlay)
    flush, flush_bytes = worker.make_flusher(fixture.x.device, L2_FLUSH_BYTES)
    for _ in range(args.warmup):
        flush()
        arm.replay()
    samples_us = []
    output = None
    for _ in range(args.iters):
        flush()
        output, elapsed_ms = arm.replay()
        samples_us.append(elapsed_ms * 1000.0)
    assert output is not None
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("benchmark output contains non-finite values")

    gpu_after = reused.selected_gpu(args.expected_gpu_uuid)
    reused.require_clock(gpu_after, args.expected_app_clock_mhz)
    artifacts, artifact_hash, cubins = _artifacts(args.jit_root)
    if artifact_hash != args.expected_jit_artifact_set_sha256:
        raise RuntimeError("measurement mutated the registered JIT artifact set")
    if cubins != [args.expected_cubin_sha256]:
        raise RuntimeError(
            f"measurement cubin drift: {cubins} != {[args.expected_cubin_sha256]}"
        )
    payload = {
        "schema": "exp016.benchmark-position.v1",
        "status": "complete",
        "arm": args.arm,
        "m": args.m,
        "fixture": args.fixture,
        "scale_kind": args.scale_kind,
        "group": args.group,
        "position": args.position,
        "abba_order": list(ABBA),
        "protocol": {
            "warmup": args.warmup,
            "iters": args.iters,
            "l2_flush_bytes": flush_bytes,
            "timing": "CUDA Graph external CUDA events",
            "process_scope": "one immutable ABBA position",
        },
        "samples_us": samples_us,
        "statistics_us": {
            "count": len(samples_us),
            "mean": statistics.fmean(samples_us),
            "median": statistics.median(samples_us),
            "p10": _quantile(samples_us, 0.10),
            "p90": _quantile(samples_us, 0.90),
            "min": min(samples_us),
            "max": max(samples_us),
            "cv": statistics.pstdev(samples_us) / statistics.fmean(samples_us),
        },
        "fixture_identity": fixture.manifest,
        "weight_identity": weights.manifest,
        "output_sha256": worker.tensor_sha256(output),
        "runtime": dict(runtime),
        "specialization": specialization,
        "gpu_after": gpu_after,
        "jit_artifacts": artifacts,
        "jit_artifact_set_sha256": artifact_hash,
        "cubin_sha256": cubins,
    }
    common.write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument(
        "--expected-baseline-sha256",
        default=EXPECTED_OVERLAY_SHA256[BASELINE],
        choices=(EXPECTED_OVERLAY_SHA256[BASELINE],),
    )
    parser.add_argument(
        "--expected-candidate-sha256",
        default=EXPECTED_OVERLAY_SHA256[CANDIDATE],
        choices=(EXPECTED_OVERLAY_SHA256[CANDIDATE],),
    )
    parser.add_argument("--jit-root", type=Path, required=True)
    parser.add_argument("--jit-policy", choices=("fresh", "reuse"), required=True)
    parser.add_argument("--expected-jit-artifact-set-sha256")
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-app-clock-mhz", type=int, required=True)
    parser.add_argument("--device-index", type=int, default=0, choices=(0,))
    parser.add_argument("--seed", type=int, default=2026, choices=(2026,))
    parser.add_argument("--output", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-case")
    validate_parser.add_argument("--m", type=int, choices=(256, 8192), required=True)
    validate_parser.add_argument("--fixture", choices=FIXTURES, required=True)
    validate_parser.add_argument("--scale-kind", choices=SCALE_KINDS, required=True)
    validate_parser.add_argument("--replays", type=int, default=2, choices=(2,))

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument(
        "--m", type=int, choices=(256, 512, 1024, 2048, 4096, 8192), required=True
    )
    measure_parser.add_argument(
        "--fixture", choices=("canonical",), default="canonical"
    )
    measure_parser.add_argument("--scale-kind", choices=SCALE_KINDS, required=True)
    measure_parser.add_argument("--group", type=int, choices=range(5), required=True)
    measure_parser.add_argument("--position", type=int, choices=range(4), required=True)
    measure_parser.add_argument("--warmup", type=int, choices=(WARMUP,), default=WARMUP)
    measure_parser.add_argument("--iters", type=int, choices=(ITERS,), default=ITERS)
    measure_parser.add_argument("--expected-cubin-sha256", required=True)
    return parser.parse_args(argv)


def _validate_case_matrix(args: argparse.Namespace) -> None:
    if args.command != "validate-case":
        return
    if args.m == 8192 and (args.fixture != "canonical" or args.scale_kind != "unequal"):
        raise RuntimeError("M8192 validation is locked to canonical unequal-scale")
    if args.fixture != "canonical" and args.m != 256:
        raise RuntimeError("hot/tail directed fixtures are locked to M256")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.flashinfer_root = args.flashinfer_root.resolve()
    args.overlay = args.overlay.resolve()
    args.jit_root = args.jit_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise RuntimeError(f"immutable output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _validate_case_matrix(args)
    _validate_registered_jit(args)
    if args.command == "measure" and args.jit_policy != "reuse":
        raise RuntimeError("measurement must reuse a correctness-validated JIT root")

    reused.BASELINE = BASELINE
    reused.CANDIDATE = CANDIDATE
    reused.ARMS = ARMS
    reused.EXPECTED_OVERLAY_SHA256 = {
        BASELINE: args.expected_baseline_sha256,
        CANDIDATE: args.expected_candidate_sha256,
    }
    source = reused.validate_source(args.flashinfer_root, args.overlay, args.arm)
    if reused.TARGET_MODULE in sys.modules:
        raise RuntimeError("target module imported before overlay installation")
    worker.install_overlay(args.overlay)
    imports = reused.configure_source_checkout(args.flashinfer_root, args.jit_root)
    if Path(imports["target_module"]) != args.overlay:
        raise RuntimeError("target module did not resolve to selected overlay")
    runtime = reused.runtime_identity(args, source)
    runtime["imports"] = imports
    runtime["harness"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": common.file_sha256(Path(__file__).resolve()),
        "reused_exp014_sha256": common.file_sha256(EXP014 / "run_exp014_arm.py"),
        "reused_exp005_sha256": common.file_sha256(EXP005 / "run_exp005_arm.py"),
    }
    if args.command == "validate-case":
        return validate_case(args, runtime)
    if args.command == "measure":
        return measure(args, runtime)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
