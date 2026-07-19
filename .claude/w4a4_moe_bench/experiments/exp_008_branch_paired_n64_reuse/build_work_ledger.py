#!/usr/bin/env python3
"""Build the CPU-safe source/descriptor work ledger for exp_008.

The ledger binds the theoretical FC1 traffic accounting to the immutable
N128/v0/v1 source identities and to the v1 producer/consumer AST structure.
It deliberately does not claim measured DRAM traffic, compiler liveness, or
dynamic Tensor work; those require the later SASS/NCU gates.
"""

import argparse
import ast
import hashlib
import json
import re
import textwrap
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

M = 128
H = 2048
I = 512
K_TILE = 128
LOGICAL_N = 128
NATIVE_N = 64
SF_VEC = 16
LOGICAL_SLICE_COUNT = I // LOGICAL_N
HALF_COUNT = LOGICAL_N // NATIVE_N
K_TRIPS = H // K_TILE

EXPECTED_SOURCE_SHA256 = {
    "anchor_8warp_n128": "3cd9e6a26056d9221f59ea6749cd601c25cbef017cf6e7349efe0925180407c1",
    "temporal_n64_v0": "1953cbb7717cda4461a4f199d05f370a4bdb35b4b8ef7556443caf36b0b12ec2",
    "branch_paired_n64_v1": "f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971",
}

EXPECTED_TOTALS = {
    "anchor_8warp_n128": {
        "a": 1048576,
        "sfa": 131072,
        "b": 1048576,
        "sfb_physical": 131072,
    },
    "temporal_n64_v0": {
        "a": 2097152,
        "sfa": 262144,
        "b": 1048576,
        "sfb_physical": 262144,
    },
    "branch_paired_n64_v1": {
        "a": 1048576,
        "sfa": 131072,
        "b": 1048576,
        "sfb_physical": 262144,
    },
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_source(path):
    raw = path.read_bytes()
    return raw, raw.decode("utf-8")


def sanitized_ast(source):
    # The workstation's system Python is 3.6; removing this future import lets
    # the otherwise-compatible source be parsed without changing the evidence.
    sanitized = re.sub(
        r"^from __future__ import annotations\s*$", "", source, flags=re.MULTILINE
    )
    return ast.parse(sanitized)


def source_region(source, start, end):
    start_at = source.find(start)
    if start_at < 0:
        raise ValueError("missing start marker: {!r}".format(start))
    end_at = source.find(end, start_at + len(start))
    if end_at < 0:
        raise ValueError("missing end marker after {!r}: {!r}".format(start, end))
    return source[start_at:end_at]


def executable_text(region):
    return "\n".join(
        line
        for line in region.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def region_identity(left, right, start, end):
    left_region = source_region(left, start, end)
    right_region = source_region(right, start, end)
    left_executable = executable_text(left_region)
    right_executable = executable_text(right_region)
    return {
        "start_marker": start,
        "end_marker": end,
        "left_sha256": sha256_bytes(left_region.encode("utf-8")),
        "right_sha256": sha256_bytes(right_region.encode("utf-8")),
        "byte_identical": left_region == right_region,
        "left_executable_sha256": sha256_bytes(left_executable.encode("utf-8")),
        "right_executable_sha256": sha256_bytes(right_executable.encode("utf-8")),
        "executable_identical": left_executable == right_executable,
    }


def target_name(target):
    return target.id if isinstance(target, ast.Name) else None


def expr_chain(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = expr_chain(node.value)
        return "{}.{}".format(prefix, node.attr) if prefix else node.attr
    if isinstance(node, ast.Call):
        return expr_chain(node.func)
    if isinstance(node, ast.Subscript):
        return expr_chain(node.value)
    return None


def root_name(node):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def call_name(node):
    return expr_chain(node.func) if isinstance(node, ast.Call) else None


def calls_named(node, name):
    return sorted(
        [item for item in ast.walk(node) if isinstance(item, ast.Call) and call_name(item) == name],
        key=lambda item: (item.lineno, item.col_offset),
    )


def assignments_named(node, name):
    matches = []
    for item in ast.walk(node):
        if isinstance(item, ast.Assign):
            if any(target_name(target) == name for target in item.targets):
                matches.append(item)
        elif isinstance(item, ast.AnnAssign) and target_name(item.target) == name:
            matches.append(item)
    return sorted(matches, key=lambda item: (item.lineno, item.col_offset))


def unique_assignment(node, name):
    matches = assignments_named(node, name)
    if len(matches) != 1:
        raise ValueError("expected one assignment to {}, found {}".format(name, len(matches)))
    return matches[0]


def assignment_value(node):
    return node.value


def ast_equal(node, expression):
    expected = ast.parse(expression, mode="eval").body
    return ast.dump(node, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def subscript_elements(node):
    if not isinstance(node, ast.Subscript):
        raise ValueError("expected Subscript, got {}".format(type(node).__name__))
    value = node.slice
    if isinstance(value, ast.Index):
        value = value.value
    if not isinstance(value, ast.Tuple):
        raise ValueError("expected tuple subscript")
    return value.elts


def find_for_loops(node, iterator):
    return sorted(
        [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.For) and target_name(item.target) == iterator
        ],
        key=lambda item: (item.lineno, item.col_offset),
    )


def tuple_first_is_fc1_half(node):
    return (
        isinstance(node, ast.Tuple)
        and len(node.elts) >= 1
        and isinstance(node.elts[0], ast.Name)
        and node.elts[0].id == "fc1_half"
    )


def build_stage_groups():
    a_per_k = M * K_TILE // 2
    b_per_branch_per_k = NATIVE_N * K_TILE // 2
    sfa_per_k = M * (K_TILE // SF_VEC)
    sfb_per_branch_per_k = LOGICAL_N * (K_TILE // SF_VEC)
    groups = []
    for logical_slice in range(LOGICAL_SLICE_COUNT):
        for half in range(HALF_COUNT):
            up_n_begin = (logical_slice * HALF_COUNT + half) * NATIVE_N
            gate_n_begin = I + up_n_begin
            groups.append(
                {
                    "logical_slice": logical_slice,
                    "half": half,
                    "k_trips": K_TRIPS,
                    "shared": {
                        "a": {
                            "bytes_per_k": a_per_k,
                            "bytes_total": a_per_k * K_TRIPS,
                        },
                        "sfa": {
                            "bytes_per_k": sfa_per_k,
                            "bytes_total": sfa_per_k * K_TRIPS,
                        },
                    },
                    "branches": {
                        "up": {
                            "global_n_range": [up_n_begin, up_n_begin + NATIVE_N],
                            "b_n64_tile_index": 2 * logical_slice + half,
                            "sfb_n128_tile_index": logical_slice,
                            "b_bytes_per_k": b_per_branch_per_k,
                            "b_bytes_total": b_per_branch_per_k * K_TRIPS,
                            "sfb_physical_bytes_per_k": sfb_per_branch_per_k,
                            "sfb_physical_bytes_total": sfb_per_branch_per_k
                            * K_TRIPS,
                            "smem_backing": ["sB_up", "sSFB_up"],
                        },
                        "gate": {
                            "global_n_range": [gate_n_begin, gate_n_begin + NATIVE_N],
                            "b_n64_tile_index": 2 * (logical_slice + LOGICAL_SLICE_COUNT)
                            + half,
                            "sfb_n128_tile_index": logical_slice
                            + LOGICAL_SLICE_COUNT,
                            "b_bytes_per_k": b_per_branch_per_k,
                            "b_bytes_total": b_per_branch_per_k * K_TRIPS,
                            "sfb_physical_bytes_per_k": sfb_per_branch_per_k,
                            "sfb_physical_bytes_total": sfb_per_branch_per_k
                            * K_TRIPS,
                            "smem_backing": ["sB", "sSFB"],
                        },
                    },
                    "tx_bytes_per_k": a_per_k
                    + sfa_per_k
                    + 2 * b_per_branch_per_k
                    + 2 * sfb_per_branch_per_k,
                }
            )
            groups[-1]["tx_bytes_total"] = groups[-1]["tx_bytes_per_k"] * K_TRIPS
    return groups


def sum_v1_totals(groups):
    return {
        "a": sum(group["shared"]["a"]["bytes_total"] for group in groups),
        "sfa": sum(group["shared"]["sfa"]["bytes_total"] for group in groups),
        "b": sum(
            branch["b_bytes_total"]
            for group in groups
            for branch in group["branches"].values()
        ),
        "sfb_physical": sum(
            branch["sfb_physical_bytes_total"]
            for group in groups
            for branch in group["branches"].values()
        ),
    }


def build_source_identity(anchor, v0, v1):
    regions = {
        "scheduler_build": (
            "        # Phase 0: cooperative init",
            "        gA = cute.local_tile(",
        ),
        "scheduler_claim_cache": (
            "        # Consumer steady state: pop one ready task per CTA",
            "            elif warp_idx < self.num_mma_warps:",
        ),
        "fc2_scatter": (
            "                    # PHASE B: Sweep ALL FC2 output tiles",
            "                    # Signal that FC2/scatter no longer needs sA",
        ),
        "fc2_producer": (
            "                    # ---- FC2 B_down loads: continuous pipeline ----",
            "                    # Ensure MMA warps finish FC2/scatter before DMA starts the",
        ),
    }
    result = {}
    for name, markers in regions.items():
        result[name] = {
            "anchor_vs_v0": region_identity(anchor, v0, markers[0], markers[1]),
            "v0_vs_v1": region_identity(v0, v1, markers[0], markers[1]),
        }
    q1_markers = (
        "                    # Both disjoint N64 activations are now durable",
        "                    # PHASE B: Sweep ALL FC2 output tiles",
    )
    result["q1"] = {
        "v0_vs_v1": region_identity(v0, v1, q1_markers[0], q1_markers[1])
    }
    return result


def build_v1_source_contract(source):
    tree = sanitized_ast(source)
    producer_region = source_region(
        source,
        "                    prod_state.reset_count()\n"
        "                    for fc1_half in cutlass.range_constexpr(2):",
        "                    # The independent phase2 pipeline aliases sB/sSFB",
    )
    producer_tree = ast.parse(textwrap.dedent(producer_region))
    producer_half_loops = find_for_loops(producer_tree, "fc1_half")
    if len(producer_half_loops) != 1:
        raise ValueError(
            "expected one producer fc1_half loop, found {}".format(
                len(producer_half_loops)
            )
        )
    producer_half = producer_half_loops[0]
    producer_k_loops = find_for_loops(producer_half, "k_tile")
    if len(producer_k_loops) != 1:
        raise ValueError(
            "expected one producer k_tile loop per half, found {}".format(
                len(producer_k_loops)
            )
        )
    producer_k = producer_k_loops[0]

    copy_pairs = Counter()
    for call in calls_named(producer_k, "cute.copy"):
        if len(call.args) < 3:
            raise ValueError("producer cute.copy missing positional operands")
        copy_pairs[(expr_chain(call.args[0]), root_name(call.args[2]))] += 1
    expected_copy_pairs = Counter(
        {
            ("tma_a", "tAsA"): 1,
            ("tma_sfa", "tAsSFA"): 1,
            ("tma_b_w13", "tBsB_w13"): 1,
            ("tma_b_w13", "tBsB_w13_up"): 1,
            ("tma_sfb_w13", "tBsSFB_w13"): 1,
            ("tma_sfb_w13", "tBsSFB_w13_up"): 1,
        }
    )

    up_index = assignment_value(unique_assignment(producer_half, "native_up_slice_idx"))
    gate_index = assignment_value(
        unique_assignment(producer_half, "native_gate_slice_idx")
    )
    sfb_gate = assignment_value(
        unique_assignment(producer_half, "tBgSFB_w13_gate_nk")
    )
    sfb_up = assignment_value(unique_assignment(producer_half, "tBgSFB_w13_up_nk"))
    sfb_gate_elements = subscript_elements(sfb_gate)
    sfb_up_elements = subscript_elements(sfb_up)

    tx_assignment = assignment_value(unique_assignment(tree, "fc1_tma_copy_bytes"))
    tx_terms = Counter()
    for call in calls_named(tx_assignment, "cute.size_in_bytes"):
        if len(call.args) != 2:
            raise ValueError("unexpected size_in_bytes arity")
        tx_terms[(expr_chain(call.args[0]), root_name(call.args[1]))] += 1
    expected_tx_terms = Counter(
        {
            ("self.a_dtype", "a_smem_one"): 1,
            ("self.b_dtype", "fc1_b_smem_one"): 2,
            ("self.sf_dtype", "sfa_smem_one"): 1,
            ("self.sf_dtype", "fc1_sfb_smem_one"): 2,
        }
    )

    ml_pipeline_value = assignment_value(unique_assignment(tree, "ml_pipeline"))
    ml_tx_keywords = [
        keyword.value
        for keyword in ml_pipeline_value.keywords
        if keyword.arg == "tx_count"
    ] if isinstance(ml_pipeline_value, ast.Call) else []

    storage_classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "StorageGated"
    ]
    if len(storage_classes) != 1:
        raise ValueError("expected one StorageGated class")
    storage_fields = {
        target_name(node.target)
        for node in storage_classes[0].body
        if isinstance(node, ast.AnnAssign)
    }

    consumer_region = source_region(
        source,
        "                    cons_state.reset_count()\n"
        "                    for fc1_half in cutlass.range_constexpr(2):",
        "                    # PHASE B: Sweep ALL FC2 output tiles",
    )
    consumer_tree = ast.parse(textwrap.dedent(consumer_region))
    consumer_half_loops = find_for_loops(consumer_tree, "fc1_half")
    if len(consumer_half_loops) != 1:
        raise ValueError(
            "expected one consumer fc1_half loop, found {}".format(
                len(consumer_half_loops)
            )
        )
    consumer_half = consumer_half_loops[0]
    k_block_loops = find_for_loops(consumer_half, "k_block_idx")
    if len(k_block_loops) != 2:
        raise ValueError(
            "expected two source k_block loop bodies, found {}".format(
                len(k_block_loops)
            )
        )

    k_loop_evidence = []
    k_loop_pairs_ok = True
    for loop in k_block_loops:
        gemms = calls_named(loop, "cute.gemm")
        observed_gemms = []
        for call in gemms:
            if len(call.args) < 4:
                raise ValueError("unexpected cute.gemm arity")
            observed_gemms.append(
                [
                    root_name(call.args[1]),
                    root_name(call.args[2]),
                    root_name(call.args[3]),
                ]
            )
        expected_gemms = [
            ["gate_acc", "tCrA_fc1", "tCrB_fc1"],
            ["up_acc", "tCrA_fc1", "tCrB_up_fc1"],
        ]
        relevant_sequence = []
        for call in sorted(
            [
                item
                for item in ast.walk(loop)
                if isinstance(item, ast.Call)
                and call_name(item) in ("mma_atom.set", "cute.gemm")
            ],
            key=lambda item: (item.lineno, item.col_offset),
        ):
            if call_name(call) == "mma_atom.set":
                if len(call.args) >= 2 and expr_chain(call.args[0]) == "WarpField.SFB":
                    relevant_sequence.append(["sfb", root_name(call.args[1])])
            else:
                relevant_sequence.append(["gemm", root_name(call.args[1])])
        expected_sequence = [
            ["sfb", "tCrSFB_fc1_half"],
            ["gemm", "gate_acc"],
            ["sfb", "tCrSFB_up_fc1_half"],
            ["gemm", "up_acc"],
        ]
        loop_ok = observed_gemms == expected_gemms and relevant_sequence == expected_sequence
        k_loop_pairs_ok = k_loop_pairs_ok and loop_ok
        k_loop_evidence.append(
            {
                "line": loop.lineno,
                "gemms": observed_gemms,
                "sfb_gemm_sequence": relevant_sequence,
                "gate_pass": loop_ok,
            }
        )

    gate_fills = calls_named(consumer_half, "gate_acc.fill")
    up_fills = calls_named(consumer_half, "up_acc.fill")
    activations = calls_named(consumer_half, "gated_activation_f32")
    stores = []
    for call in calls_named(consumer_half, "cute.copy"):
        if (
            len(call.args) >= 3
            and root_name(call.args[0]) == "fc1_tiled_copy_r2s"
            and root_name(call.args[2]) == "fc1_tRS_sD"
        ):
            stores.append(call)
    gemm_lines = [call.lineno for loop in k_block_loops for call in calls_named(loop, "cute.gemm")]
    immediate_activation_store = (
        len(activations) == 1
        and len(stores) == 1
        and bool(gemm_lines)
        and max(gemm_lines) < activations[0].lineno < stores[0].lineno
    )

    q1_assignments = assignments_named(consumer_tree, "sA_u8")
    half_node_ids = {id(node) for node in ast.walk(consumer_half)}
    q1_outside_half = (
        len(q1_assignments) == 1 and id(q1_assignments[0]) not in half_node_ids
    )

    full_mma_n = assignment_value(unique_assignment(consumer_half, "full_mma_n"))
    local_tile_checks = {}
    for name, backing in (
        ("sSFB_fc1_half", "sSFB_fc1"),
        ("sSFB_up_fc1_half", "sSFB_up_fc1"),
    ):
        value = assignment_value(unique_assignment(consumer_half, name))
        local_tile_checks[name] = (
            isinstance(value, ast.Call)
            and call_name(value) == "cute.local_tile"
            and len(value.args) >= 3
            and root_name(value.args[0]) == backing
            and tuple_first_is_fc1_half(value.args[2])
        )

    checks = {
        "producer_one_half_loop": len(producer_half_loops) == 1,
        "producer_one_k_loop_per_half": len(producer_k_loops) == 1,
        "producer_exact_six_copy_multiset": copy_pairs == expected_copy_pairs,
        "producer_one_acquire_per_k": len(
            calls_named(producer_k, "ml_pipeline.producer_acquire")
        )
        == 1,
        "producer_one_commit_per_k": len(
            calls_named(producer_k, "ml_pipeline.producer_commit")
        )
        == 1,
        "producer_one_advance_per_k": len(calls_named(producer_k, "prod_state.advance"))
        == 1,
        "producer_up_b_index_formula": ast_equal(
            up_index, "intermediate_slice * Int32(2) + Int32(fc1_half)"
        ),
        "producer_gate_b_index_formula": ast_equal(
            gate_index,
            "(intermediate_slice + gate_tile_cnt) * Int32(2) + Int32(fc1_half)",
        ),
        "producer_up_sfb_index_formula": len(sfb_up_elements) == 4
        and ast_equal(sfb_up_elements[1], "intermediate_slice"),
        "producer_gate_sfb_index_formula": len(sfb_gate_elements) == 4
        and ast_equal(sfb_gate_elements[1], "intermediate_slice + gate_tile_cnt"),
        "tx_count_exact_six_term_multiset": tx_terms == expected_tx_terms,
        "pipeline_uses_fc1_tx_count": len(ml_tx_keywords) == 1
        and isinstance(ml_tx_keywords[0], ast.Name)
        and ml_tx_keywords[0].id == "fc1_tma_copy_bytes",
        "independent_up_smem_backings": {"sB_up", "sSFB_up"}.issubset(
            storage_fields
        ),
        "independent_up_tma_partitions": source.count(
            "tBsB_w13_up, _tBgB_w13_up = cpasync.tma_partition("
        )
        == 1
        and source.count(
            "tBsSFB_w13_up, _tBgSFB_w13_up = cpasync.tma_partition("
        )
        == 1,
        "consumer_one_half_loop": len(consumer_half_loops) == 1,
        "consumer_two_k_loop_bodies": len(k_block_loops) == 2,
        "consumer_gate_up_paired_in_each_k_loop": k_loop_pairs_ok,
        "consumer_accumulators_cleared_once_per_half": len(gate_fills) == 1
        and len(up_fills) == 1,
        "consumer_gate_up_sfb_select_same_half": all(local_tile_checks.values()),
        "consumer_immediate_activation_store_inside_half": immediate_activation_store,
        "consumer_half_store_offset_formula": ast_equal(
            full_mma_n, "fc1_half * fc1_n_tiles + mma_n"
        ),
        "q1_once_outside_half": q1_outside_half,
        "one_gate_n64_accumulator_object": source.count(
            "gate_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)"
        )
        == 1,
        "one_up_n64_accumulator_object": source.count(
            "up_acc = cute.make_rmem_tensor(fc1_acc_shape, self.acc_dtype)"
        )
        == 1,
        "single_fc1_to_fc2_handoff": source.count(
            "self.pass_gate_barrier.arrive_unaligned()"
        )
        == 1
        and source.count("self.pass_gate_barrier.wait_unaligned()") == 1,
    }
    return {
        "producer_region_sha256": sha256_bytes(producer_region.encode("utf-8")),
        "consumer_q1_region_sha256": sha256_bytes(
            consumer_region.encode("utf-8")
        ),
        "producer_copy_multiset": {
            "{} -> {}".format(key[0], key[1]): value
            for key, value in sorted(copy_pairs.items())
        },
        "tx_count_term_multiset": {
            "{} / {}".format(key[0], key[1]): value
            for key, value in sorted(tx_terms.items())
        },
        "consumer_k_loop_evidence": k_loop_evidence,
        "consumer_sfb_half_views": local_tile_checks,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "evidence_boundary": (
            "Source AST/order proves the registered implementation contract only; "
            "it does not reconstruct compiler register liveness or dynamic issue."
        ),
    }


def build_payload(paths):
    sources = {}
    identities = {}
    for arm, path in paths.items():
        raw, source = read_source(path)
        digest = sha256_bytes(raw)
        sources[arm] = source
        identities[arm] = {
            "path": str(path),
            "sha256": digest,
            "expected_sha256": EXPECTED_SOURCE_SHA256[arm],
            "hash_locked": digest == EXPECTED_SOURCE_SHA256[arm],
            "ast_parse_pass": True,
        }
        sanitized_ast(source)

    groups = build_stage_groups()
    v1_totals = sum_v1_totals(groups)
    source_identity = build_source_identity(
        sources["anchor_8warp_n128"],
        sources["temporal_n64_v0"],
        sources["branch_paired_n64_v1"],
    )
    source_contract = build_v1_source_contract(sources["branch_paired_n64_v1"])

    observed_up_ranges = sorted(
        tuple(group["branches"]["up"]["global_n_range"]) for group in groups
    )
    observed_gate_ranges = sorted(
        tuple(group["branches"]["gate"]["global_n_range"]) for group in groups
    )
    expected_up_ranges = [(n, n + NATIVE_N) for n in range(0, I, NATIVE_N)]
    expected_gate_ranges = [
        (n, n + NATIVE_N) for n in range(I, 2 * I, NATIVE_N)
    ]
    stage_keys = [(group["logical_slice"], group["half"]) for group in groups]
    source_identity_checks = {}
    for name, comparisons in source_identity.items():
        for comparison, evidence in comparisons.items():
            source_identity_checks[
                "{}_{}_executable_identical".format(name, comparison)
            ] = evidence["executable_identical"]

    checks = {
        "all_three_source_hashes_locked": all(
            identity["hash_locked"] for identity in identities.values()
        ),
        "all_three_sources_ast_parse": all(
            identity["ast_parse_pass"] for identity in identities.values()
        ),
        "eight_unique_slice_half_stage_groups": len(groups) == 8
        and len(set(stage_keys)) == 8,
        "up_n64_ranges_exact_once": observed_up_ranges == expected_up_ranges,
        "gate_n64_ranges_exact_once": observed_gate_ranges == expected_gate_ranges,
        "each_stage_tx_count_is_19456": all(
            group["tx_bytes_per_k"] == 19456 for group in groups
        ),
        "v1_traffic_totals_exact": v1_totals
        == EXPECTED_TOTALS["branch_paired_n64_v1"],
        "v1_a_sfa_equal_anchor": all(
            v1_totals[name] == EXPECTED_TOTALS["anchor_8warp_n128"][name]
            for name in ("a", "sfa")
        ),
        "v1_a_sfa_are_half_v0": all(
            2 * v1_totals[name] == EXPECTED_TOTALS["temporal_n64_v0"][name]
            for name in ("a", "sfa")
        ),
        "v1_b_equal_anchor_and_v0": v1_totals["b"]
        == EXPECTED_TOTALS["anchor_8warp_n128"]["b"]
        == EXPECTED_TOTALS["temporal_n64_v0"]["b"],
        "v1_physical_sfb_equal_v0_and_2x_anchor": v1_totals["sfb_physical"]
        == EXPECTED_TOTALS["temporal_n64_v0"]["sfb_physical"]
        == 2 * EXPECTED_TOTALS["anchor_8warp_n128"]["sfb_physical"],
        "v1_source_structure_contract": source_contract["gate_pass"],
    }
    checks.update(source_identity_checks)

    payload = {
        "schema": "exp008.work-ledger.v1",
        "identity": identities,
        "shape": {
            "M": M,
            "H": H,
            "I_tp": I,
            "logical_n": LOGICAL_N,
            "native_fc1_n": NATIVE_N,
            "k_tile": K_TILE,
            "k_trips": K_TRIPS,
        },
        "stage_groups": groups,
        "traffic_totals": {
            "anchor_8warp_n128": EXPECTED_TOTALS["anchor_8warp_n128"],
            "temporal_n64_v0": EXPECTED_TOTALS["temporal_n64_v0"],
            "branch_paired_n64_v1": v1_totals,
            "boundary": "one complete Gate+Up FC1 sweep across I_tp=512",
            "semantics": "descriptor payload bytes, not measured DRAM traffic",
        },
        "source_identity": source_identity,
        "source_structure": source_contract,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "evidence_boundary": [
            "The ledger proves immutable source identity, registered coordinates, descriptor payload accounting, and source structure.",
            "SASS separately proves static OMMA/spill; matched NCU separately proves executed Tensor work and dynamic spill.",
            "Descriptor payload bytes are not a measurement of L2 or DRAM traffic.",
        ],
    }
    payload["evidence_sha256"] = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return payload


def write_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-n128",
        type=Path,
        default=RESULTS / "overlays/anchor_8warp_n128/moe_dynamic_kernel.py",
    )
    parser.add_argument(
        "--temporal-v0",
        type=Path,
        default=RESULTS / "overlays/temporal_n64_v0/moe_dynamic_kernel.py",
    )
    parser.add_argument(
        "--candidate-v1",
        type=Path,
        default=RESULTS / "overlays/branch_paired_n64_v1/moe_dynamic_kernel.py",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "work_ledger.json"
    )
    args = parser.parse_args()
    paths = {
        "anchor_8warp_n128": args.anchor_n128.resolve(),
        "temporal_n64_v0": args.temporal_v0.resolve(),
        "branch_paired_n64_v1": args.candidate_v1.resolve(),
    }
    try:
        payload = build_payload(paths)
    except Exception as error:
        payload = {
            "schema": "exp008.work-ledger.v1",
            "gate_pass": False,
            "error": "{}: {}".format(type(error).__name__, error),
            "paths": {name: str(path) for name, path in paths.items()},
        }
        write_payload(args.output.resolve(), payload)
        print(json.dumps(payload, sort_keys=True))
        return 2
    write_payload(args.output.resolve(), payload)
    print(json.dumps({"gate_pass": payload["gate_pass"], "checks": payload["checks"]}, sort_keys=True))
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
