#!/usr/bin/env python3
"""Build the exp_023 role-preflight or two-stage candidate overlay.

The builder is deliberately identity locked to the current Opt source.  The
``role-preflight`` variant is a throw-away first shutter: it proves that a
13-warp CTA can keep the four Scatter warps out of all MMA fragment setup
before the stage protocol is introduced.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent.parent / "moe_dynamic_kernel_opt.py"
EXPECTED_BASELINE_SHA256 = (
    "ad4c26f9f808586e3204e7d495b6c439175f708d3713d9ab61b330848fbf8d19"
)
BASELINE_GIT_BLOB = "0aa1cf9bb4b345c9438e3a5f0c62b56c1f9b046b"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_locked_baseline() -> tuple[bytes, str]:
    live_bytes = BASELINE.read_bytes()
    if sha256(live_bytes) == EXPECTED_BASELINE_SHA256:
        return live_bytes, "live pre-integration source"
    repo = ROOT.parents[3]
    baseline_bytes = subprocess.check_output(
        ["git", "cat-file", "blob", BASELINE_GIT_BLOB], cwd=str(repo)
    )
    if sha256(baseline_bytes) != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("locked baseline git blob SHA drift")
    return baseline_bytes, "locked git blob " + BASELINE_GIT_BLOB


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, got {count}")
    return text.replace(old, new)


def replace_span(text: str, start: str, end: str, replacement: str, label: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(
            f"{label}: non-unique anchors ({text.count(start)}, {text.count(end)})"
        )
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "".join(
        prefix + line if line.strip() else line for line in text.splitlines(True)
    )


def specialize_role_loops(candidate: str) -> str:
    """Keep all MMA values and their consumer loop inside the W0-W7 region."""
    setup_anchor = """        if warp_idx < self.num_mma_warps:
            # These slices and every derived fragment are role-local: W8-W12
"""
    setup_end_anchor = """        # ===================================================================
        # Per-warp setup for the consumer steady state
"""
    main_anchor = """        consumer_live = Int32(1)
        while consumer_live > Int32(0):
"""
    math_anchor = "            elif warp_idx < self.num_mma_warps:\n"
    scatter_anchor = "            elif warp_idx < self.tma_load_warp_id:\n"
    tma_anchor = "            elif warp_idx == self.tma_load_warp_id:\n"
    tail_anchor = """        if warp_idx == self.tma_load_warp_id:
            ml_pipeline.producer_tail(prod_state)
"""
    return_anchor = "        return\n"
    for label, anchor in (
        ("setup", setup_anchor),
        ("setup end", setup_end_anchor),
        ("main", main_anchor),
        ("math", math_anchor),
        ("scatter", scatter_anchor),
        ("tma", tma_anchor),
        ("tail", tail_anchor),
    ):
        if candidate.count(anchor) != 1:
            raise RuntimeError(
                f"role specialization {label}: got {candidate.count(anchor)}"
            )

    setup_start = candidate.index(setup_anchor)
    setup_line_end = setup_start + setup_anchor.index("\n") + 1
    setup_finish = candidate.index(setup_end_anchor, setup_start)
    setup_body = candidate[setup_line_end:setup_finish]

    main_start = candidate.index(main_anchor, setup_finish)
    math_start = candidate.index(math_anchor, main_start)
    scatter_start = candidate.index(scatter_anchor, math_start)
    tma_start = candidate.index(tma_anchor, scatter_start)
    tail_start = candidate.index(tail_anchor, tma_start)
    return_start = candidate.index(return_anchor, tail_start)

    loop_prefix = candidate[main_start:math_start]
    math_body = candidate[math_start + len(math_anchor) : scatter_start]
    scatter_body = candidate[scatter_start + len(scatter_anchor) : tma_start]
    tma_body = candidate[tma_start + len(tma_anchor) : tail_start]
    tail_body_start = tail_start + tail_anchor.index("\n") + 1
    tail_body = candidate[tail_body_start:return_start]

    role_code = (
        "        if warp_idx < self.num_mma_warps:\n"
        + setup_body
        + indent(loop_prefix, 4)
        + "                else:\n"
        + indent(math_body, 4)
        + "        elif warp_idx < self.tma_load_warp_id:\n"
        + indent(loop_prefix, 4)
        + "                else:\n"
        + indent(scatter_body, 4)
        + "        elif warp_idx == self.tma_load_warp_id:\n"
        + "            prod_state = pipeline.make_pipeline_state(\n"
        + "                pipeline.PipelineUserType.Producer, self.ab_stage\n"
        + "            )\n"
        + "            phase2_prod_state = pipeline.make_pipeline_state(\n"
        + "                pipeline.PipelineUserType.Producer, self.ab_stage\n"
        + "            )\n"
        + indent(loop_prefix, 4)
        + "                else:\n"
        + indent(tma_body, 4)
        + tail_body
    )
    return candidate[:setup_start] + role_code + candidate[return_start:]


def build_role_preflight(baseline: str) -> str:
    candidate = baseline
    candidate = replace_once(
        candidate,
        """        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
""",
        """        self.num_mma_warps = 8
        self.num_scatter_warps = 4
        self.scatter_warp_begin = self.num_mma_warps
        self.tma_load_warp_id = self.scatter_warp_begin + self.num_scatter_warps
        self.num_threads_per_warp = 32
        self.route_num_warps = self.num_mma_warps + 1
        self.route_threads_per_cta = self.route_num_warps * self.num_threads_per_warp
        self.threads_per_cta = (self.tma_load_warp_id + 1) * self.num_threads_per_warp
""",
        "warp roles",
    )
    candidate = replace_once(
        candidate,
        """        self.load_register_requirement = 32
        self.mma_register_requirement = 232
""",
        """        self.load_register_requirement = 32
        self.scatter_register_requirement = 48
        self.mma_register_requirement = 224
""",
        "register budgets",
    )

    candidate = replace_once(
        candidate,
        """        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)
""",
        """        # W0-W8 retain the exact original Route/Q0 virtual mapping.
        # W9-W12 still execute every CTA-wide synchronization, but a sentinel
        # virtual id keeps them out of all flat work and route-cache indexing.
        route_warp_idx = Int32(warp_idx)
        flat_tid = Int32(bidz) * Int32(self.route_threads_per_cta) + Int32(tidx)
        if warp_idx >= Int32(self.route_num_warps):
            # Override *after* the bidz arithmetic: adding bidz to INT32_MAX
            # would wrap negative for CTAs 1..109.  The route-warp sentinel is
            # also bounded so batch_base + sentinel cannot overflow.
            flat_tid = Int32(0x40000000)
            route_warp_idx = Int32(0x00100000)
        flat_stride = Int32(gdim_z) * Int32(self.route_threads_per_cta)
""",
        "route virtual mapping",
    )
    candidate = candidate.replace(
        "token_idx = batch_base + warp_idx", "token_idx = batch_base + route_warp_idx"
    )
    if candidate.count("token_idx = batch_base + route_warp_idx") != 2:
        raise RuntimeError("route token ownership patch did not apply twice")
    candidate = candidate.replace(
        "route_slot_base = warp_idx * Int32(32)",
        "route_slot_base = route_warp_idx * Int32(32)",
    )
    if candidate.count("route_slot_base = route_warp_idx * Int32(32)") != 2:
        raise RuntimeError("route cache ownership patch did not apply twice")

    candidate = replace_once(
        candidate,
        """        thr_mma = tiled_mma.get_slice(tidx)
        fc1_thr_mma = fc1_tiled_mma.get_slice(tidx)

""",
        "",
        "remove common MMA slices",
    )

    setup_start = (
        "        # FC2 fragment partitions retain the original N128 contract.\n"
    )
    setup_end = "        # ===================================================================\n        # Per-warp setup for the consumer steady state\n"
    begin = candidate.index(setup_start)
    finish = candidate.index(setup_end, begin)
    setup = candidate[begin:finish]
    common_counts = """        # Counts needed by all three role loops.  They do not construct
        # any MMA thread slice or fragment for W8-W12.
        k_tile_cnt = cute.size(gA, mode=[3])
        fc1_k_tile_cnt = k_tile_cnt
        native_fc1_tile_cnt = cute.size(gB_w13_tiled, mode=[2]) // Int32(2)
        gate_tile_cnt = native_fc1_tile_cnt // Int32(2)
        output_tile_cnt = cute.size(gB_down, mode=[2])

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
        elif warp_idx < self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.scatter_register_requirement)
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx < self.num_mma_warps:
            # These slices and every derived fragment are role-local: W8-W12
            # never execute or inherit the MMA setup path.
            thr_mma = tiled_mma.get_slice(tidx)
            fc1_thr_mma = fc1_tiled_mma.get_slice(tidx)
"""
    candidate = (
        candidate[:begin] + common_counts + indent(setup, 4) + candidate[finish:]
    )

    candidate = replace_once(
        candidate,
        """        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
""",
        """        # Register redistribution is issued before role-local setup above.
""",
        "remove late register redistribution",
    )

    tma_branch = "            elif warp_idx == self.tma_load_warp_id:\n"
    if candidate.count(tma_branch) != 1:
        raise RuntimeError("expected one steady-state TMA role branch")
    idle_scatter = """            elif warp_idx < self.tma_load_warp_id:
                # Role-preflight only: W8-W11 deliberately execute no MMA
                # setup and no Scatter work.  They join the two full-CTA
                # ownership handoffs so the original producer/TMA protocol
                # remains launchable with a 416-thread CTA.
                task_slice_count_val = _ld_shared_i32(ctrl_base_addr + Int32(20))
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    self.pass_gate_barrier.arrive_unaligned()
                    self.pass_final_barrier.arrive_unaligned()
                    slice_idx += Int32(1)

"""
    candidate = candidate.replace(tma_branch, idle_scatter + tma_branch)
    candidate = specialize_role_loops(candidate)

    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("role-preflight",), default="role-preflight"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_bytes, baseline_origin = read_locked_baseline()
    observed = sha256(baseline_bytes)
    if observed != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"baseline hash drift: {observed}")
    baseline = baseline_bytes.decode()
    route_last_active_flat = 109 * 288 + 287
    if route_last_active_flat != 31679:
        raise AssertionError("route virtual mapping known-answer drift")
    if 0x40000000 + 110 * 288 >= 0x7FFFFFFF:
        raise AssertionError("route flat-work sentinel can overflow Int32")
    if 0x00100000 + 8192 >= 0x7FFFFFFF:
        raise AssertionError("route warp sentinel can overflow token_idx")
    candidate = build_role_preflight(baseline)
    default_output = Path("/tmp/exp023_role_preflight/moe_dynamic_kernel.py")

    output = (args.output or default_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(candidate, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="moe_dynamic_kernel_opt.py",
            tofile=f"exp023/{args.variant}/moe_dynamic_kernel.py",
            n=2,
        )
    )
    output.with_suffix(".diff").write_text(diff, encoding="utf-8")
    payload = {
        "schema": "exp023.kernel-overlay-identity.v1",
        "variant": args.variant,
        "baseline": str(BASELINE),
        "baseline_origin": baseline_origin,
        "baseline_sha256": observed,
        "candidate": str(output),
        "candidate_sha256": sha256(candidate.encode()),
        "ephemeral": args.variant == "role-preflight",
        "route_virtual_mapping_known_answers": {
            "bidz109_tidx287": route_last_active_flat,
            "extra_warp_flat_sentinel": 0x40000000,
            "extra_warp_route_index_sentinel": 0x00100000,
        },
    }
    output.with_suffix(".identity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
