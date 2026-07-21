#!/usr/bin/env python3
"""Fail-closed AST gate for the dynamic-MoE phase skeleton.

This gate intentionally checks source structure only.  It does not claim that
CuTeDSL will inline the helpers, preserve resources, or produce correct GPU
results; exp_015's compile, SASS, correctness, and performance gates cover
those properties separately.
"""

import argparse
import ast
import sys
from pathlib import Path


REQUIRED_HELPERS = (
    "resident_grid_barrier",
    "publish_ready_tasks",
    "publish_deferred_tasks",
    "claim_and_cache_task",
    "fc1_gate_up_swiglu_to_sC",
    "quantize_q1_sC_to_sA_sSFA",
    "load_fc2_a_fragments",
    "fc2_to_sC",
    "scatter_sC_to_gmem",
    "load_fc1_tma_slice",
    "load_fc2_tma_slice",
    "initialize_route_q0_and_publish",
)

# A helper that advances a pipeline state must return that exact state, and
# __call__ must explicitly rebind it at the call site.
PIPELINE_STATE_HELPERS = {
    "fc1_gate_up_swiglu_to_sC": "cons_state",
    "fc2_to_sC": "phase2_cons_state",
    "load_fc1_tma_slice": "prod_state",
    "load_fc2_tma_slice": "phase2_prod_state",
}

ACCUMULATOR_OWNERS = {
    "gate_acc": "fc1_gate_up_swiglu_to_sC",
    "up_acc": "fc1_gate_up_swiglu_to_sC",
    "down_acc": "fc2_to_sC",
}


def dotted_name(node):
    """Return a dotted AST name such as self.foo.bar, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def iter_nodes(root):
    """Walk one function body without descending into nested definitions."""
    stack = list(reversed(getattr(root, "body", [])))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        children = list(ast.iter_child_nodes(node))
        stack.extend(reversed(children))


def method_calls(method):
    return [node for node in iter_nodes(method) if isinstance(node, ast.Call)]


def is_cute_jit(method):
    return any(dotted_name(item) == "cute.jit" for item in method.decorator_list)


def is_cute_kernel(method):
    return any(dotted_name(item) == "cute.kernel" for item in method.decorator_list)


def literal_value(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def direct_call(statement):
    """Return a call made directly by an Expr/Assign statement, if any."""
    value = None
    if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)):
        value = statement.value
    if isinstance(value, ast.Call):
        return value
    return None


def assignment_target_name(statement):
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
    elif isinstance(statement, ast.AnnAssign):
        target = statement.target
    else:
        return None
    if isinstance(target, ast.Name):
        return target.id
    return None


def is_name_return(node, name):
    return isinstance(node.value, ast.Name) and node.value.id == name


def call_uses_name(call, name):
    return any(
        isinstance(node, ast.Name) and node.id == name for node in ast.walk(call)
    )


def root_name(node):
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def is_index(node, base, index):
    if not isinstance(node, ast.Subscript) or dotted_name(node.value) != base:
        return False
    slice_node = node.slice
    if isinstance(slice_node, ast.Index):
        slice_node = slice_node.value
    return literal_value(slice_node) == index


def assigned_values(method, target_path):
    values = []
    for node in iter_nodes(method):
        if isinstance(node, ast.Assign):
            if any(dotted_name(target) == target_path for target in node.targets):
                values.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if dotted_name(node.target) == target_path:
                values.append(node.value)
    return values


def is_epi_rest_expression(node):
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.FloorDiv)
        and is_index(node.left, "self.tile_shape_mnk", 0)
        and is_index(node.right, "self.epi_tile", 0)
    )


def is_fc1_final_half_test(test):
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    return (
        isinstance(test.left, ast.Name)
        and test.left.id == "fc1_half"
        and literal_value(test.comparators[0]) == 1
    )


def tuple_binding_from_parameter(method, bound_name, parameter_hint):
    """Prove that bound_name is unpacked from a meaningfully named argument."""
    parameters = {arg.arg for arg in method.args.args}
    for node in iter_nodes(method):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, (ast.Tuple, ast.List)):
            continue
        target_names = {item.id for item in target.elts if isinstance(item, ast.Name)}
        if bound_name not in target_names or not isinstance(node.value, ast.Name):
            continue
        source_name = node.value.id
        if source_name in parameters and parameter_hint in source_name:
            return True
    return False


def fence_is_cta_shared(call):
    if dotted_name(call.func) != "cute.arch.fence_proxy":
        return False
    if not call.args or literal_value(call.args[0]) != "async.shared":
        return False
    spaces = [
        literal_value(keyword.value)
        for keyword in call.keywords
        if keyword.arg == "space"
    ]
    return spaces == ["cta"]


def enclosing_methods(class_node):
    methods = {}
    duplicates = set()
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in methods:
                duplicates.add(node.name)
            methods[node.name] = node
    return methods, duplicates


def parent_map(root):
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def validate_source(source, filename="<source>"):
    """Return structural violations.  An empty list means PASS."""
    errors = []
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError, TypeError) as exc:
        return ["AST parse failed (fail closed): {}".format(exc)]

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MoEDynamicKernel"
    ]
    if len(classes) != 1:
        return [
            "expected exactly one top-level MoEDynamicKernel class; found {}".format(
                len(classes)
            )
        ]
    kernel_class = classes[0]
    methods, duplicate_methods = enclosing_methods(kernel_class)
    if duplicate_methods:
        errors.append(
            "duplicate method definitions: {}".format(
                ", ".join(sorted(duplicate_methods))
            )
        )

    for name in REQUIRED_HELPERS + ("__call__",):
        method = methods.get(name)
        if method is None:
            errors.append("missing required helper/method: {}".format(name))
        elif not is_cute_jit(method):
            errors.append("{} must be decorated with @cute.jit".format(name))

    kernel = methods.get("kernel")
    if kernel is None:
        errors.append("missing required caller method: kernel")
    elif not is_cute_kernel(kernel):
        errors.append("kernel must be decorated with @cute.kernel")

    if "fc2_scatter_to_gmem" in methods:
        errors.append(
            "legacy combined helper fc2_scatter_to_gmem is forbidden; "
            "use fc2_to_sC + scatter_sC_to_gmem"
        )

    # A generic ctx hides phase ownership.  Ban the exact identifier in the
    # whole candidate so a context bundle cannot silently reappear elsewhere.
    ctx_locations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.arg)
            and node.arg == "ctx"
            or isinstance(node, ast.Name)
            and node.id == "ctx"
        ):
            ctx_locations.append(getattr(node, "lineno", 0))
    if ctx_locations:
        errors.append(
            "generic identifier 'ctx' is forbidden (lines {})".format(
                ", ".join(str(line) for line in sorted(set(ctx_locations)))
            )
        )

    for method in methods.values():
        if (
            (is_cute_jit(method) or is_cute_kernel(method))
            and method.name.startswith("_")
            and not (method.name.startswith("__") and method.name.endswith("__"))
        ):
            errors.append(
                "@cute.jit phase/helper must not have a leading underscore: {}".format(
                    method.name
                )
            )

    if any(name not in methods for name in REQUIRED_HELPERS + ("__call__", "kernel")):
        return errors

    caller = methods["kernel"]

    # Pipeline state is a data dependency, not hidden mutable context.
    for helper_name, state_name in sorted(PIPELINE_STATE_HELPERS.items()):
        helper = methods[helper_name]
        helper_nodes = list(iter_nodes(helper))
        advances = [
            call
            for call in method_calls(helper)
            if dotted_name(call.func) == state_name + ".advance"
        ]
        if not advances:
            errors.append("{} must visibly advance {}".format(helper_name, state_name))

        returns = [node for node in helper_nodes if isinstance(node, ast.Return)]
        if len(returns) != 1 or not is_name_return(returns[0], state_name):
            errors.append(
                "{} must have one explicit `return {}`".format(helper_name, state_name)
            )

        call_sites = []
        for statement in iter_nodes(caller):
            call = direct_call(statement)
            if call is not None and dotted_name(call.func) == "self." + helper_name:
                call_sites.append((statement, call))
        if len(call_sites) != 1:
            errors.append(
                "kernel must contain exactly one syntactic call site for {}; found {}".format(
                    helper_name, len(call_sites)
                )
            )
        elif assignment_target_name(call_sites[0][0]) != state_name:
            errors.append(
                "kernel must explicitly rebind `{0} = self.{1}(...)`".format(
                    state_name, helper_name
                )
            )

    # If a required helper advances some other simple-name state, it has not
    # been declared in the return/rebind contract above.
    state_helpers = set(PIPELINE_STATE_HELPERS)
    for helper_name in REQUIRED_HELPERS:
        helper = methods[helper_name]
        for call in method_calls(helper):
            if isinstance(call.func, ast.Attribute) and call.func.attr == "advance":
                receiver = dotted_name(call.func.value)
                if helper_name not in state_helpers:
                    errors.append(
                        "{} advances hidden state {}; add an explicit return/rebind contract".format(
                            helper_name, receiver or "<dynamic>"
                        )
                    )
                elif receiver != PIPELINE_STATE_HELPERS[helper_name]:
                    errors.append(
                        "{} advances unexpected state {}; expected {}".format(
                            helper_name,
                            receiver or "<dynamic>",
                            PIPELINE_STATE_HELPERS[helper_name],
                        )
                    )

    # Accumulators must be born, consumed, and die in their owning phase.
    for accumulator, owner in sorted(ACCUMULATOR_OWNERS.items()):
        users = []
        for method_name, method in methods.items():
            if any(
                isinstance(node, ast.Name) and node.id == accumulator
                for node in iter_nodes(method)
            ):
                users.append(method_name)
        if users != [owner]:
            errors.append(
                "{} must be local only to {}; observed in {}".format(
                    accumulator, owner, ", ".join(sorted(users)) or "no method"
                )
            )
            continue

        owner_method = methods[owner]
        allocations = []
        for statement in iter_nodes(owner_method):
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id == accumulator
                and isinstance(statement.value, ast.Call)
                and dotted_name(statement.value.func) == "cute.make_rmem_tensor"
            ):
                allocations.append(statement)
        if len(allocations) != 1:
            errors.append(
                "{} must have one visible cute.make_rmem_tensor allocation in {}".format(
                    accumulator, owner
                )
            )

        gemm_uses = [
            call
            for call in method_calls(owner_method)
            if dotted_name(call.func) == "cute.gemm"
            and call_uses_name(call, accumulator)
        ]
        if not gemm_uses:
            errors.append(
                "{} must be consumed by cute.gemm in {}".format(accumulator, owner)
            )

    fc2 = methods["fc2_to_sC"]
    scatter = methods["scatter_sC_to_gmem"]
    fc2_argument_names = [arg.arg for arg in fc2.args.args]
    if "sC" not in fc2_argument_names and not tuple_binding_from_parameter(
        fc2, "sC", "storage"
    ):
        errors.append(
            "fc2_to_sC must receive sC explicitly or unpack it from a named storage argument"
        )
    scatter_argument_names = [arg.arg for arg in scatter.args.args]
    if "sC" not in scatter_argument_names:
        errors.append("scatter_sC_to_gmem must receive sC explicitly")

    scatter_reads_sC = any(
        isinstance(node, ast.Subscript)
        and root_name(node.value) == "sC"
        and isinstance(node.ctx, ast.Load)
        for node in iter_nodes(scatter)
    )
    if not scatter_reads_sC:
        errors.append("scatter_sC_to_gmem must visibly read from sC")

    # Locate the one enclosing per-slice loop.  We use AST source order only
    # after proving that every phase call occurs exactly once in that loop.
    phase_paths = {
        "self.fc1_gate_up_swiglu_to_sC": "fc1",
        "self.quantize_q1_sC_to_sA_sSFA": "q1",
        "self.load_fc2_a_fragments": "load_fc2_a",
        "self.fc2_to_sC": "fc2",
        "self.scatter_sC_to_gmem": "scatter",
        "self.pass_final_barrier.arrive_unaligned": "pass_final",
    }
    candidate_loops = []
    for node in iter_nodes(caller):
        if not isinstance(node, ast.While):
            continue
        labels = [
            phase_paths[dotted_name(call.func)]
            for call in method_calls(node)
            if dotted_name(call.func) in phase_paths
        ]
        if set(labels) == set(phase_paths.values()):
            candidate_loops.append(node)
    # The outer persistent-task loop also lexically contains the per-slice
    # loop.  Select the innermost loop that still contains the complete phase
    # chain; nested FC2 tile loops do not contain FC1/Q1 and therefore do not
    # qualify.
    innermost_loops = []
    for loop in candidate_loops:
        descendants = set(iter_nodes(loop))
        if not any(
            other is not loop and other in descendants for other in candidate_loops
        ):
            innermost_loops.append(loop)
    if len(innermost_loops) != 1:
        errors.append(
            "expected one per-slice loop containing all math phases; found {}".format(
                len(innermost_loops)
            )
        )
    else:
        phase_loop = innermost_loops[0]
        ordered_calls = sorted(
            method_calls(phase_loop),
            key=lambda call: (
                getattr(call, "lineno", 0),
                getattr(call, "col_offset", 0),
            ),
        )
        phase_events = [
            (phase_paths[dotted_name(call.func)], call)
            for call in ordered_calls
            if dotted_name(call.func) in phase_paths
        ]
        labels = [label for label, unused_call in phase_events]
        expected = [
            "fc1",
            "q1",
            "load_fc2_a",
            "fc2",
            "scatter",
            "pass_final",
        ]
        if labels != expected:
            errors.append(
                "caller phase order must be {}; observed {}".format(
                    " -> ".join(expected), " -> ".join(labels) or "none"
                )
            )
        else:
            positions = {
                label: (call.lineno, call.col_offset) for label, call in phase_events
            }
            fences = [call for call in ordered_calls if fence_is_cta_shared(call)]
            malformed_fences = [
                call
                for call in ordered_calls
                if dotted_name(call.func) == "cute.arch.fence_proxy"
                and not fence_is_cta_shared(call)
            ]
            if malformed_fences:
                errors.append(
                    "all caller phase fences must be fence_proxy('async.shared', space='cta')"
                )
            epilog_syncs = [
                call
                for call in ordered_calls
                if dotted_name(call.func) == "self.epilog_sync_barrier.arrive_and_wait"
            ]

            boundaries = (("fc1", "q1"), ("q1", "fc2"), ("fc2", "scatter"))
            for left, right in boundaries:
                left_pos = positions[left]
                right_pos = positions[right]
                boundary_fences = [
                    call
                    for call in fences
                    if left_pos < (call.lineno, call.col_offset) < right_pos
                ]
                boundary_barriers = [
                    call
                    for call in epilog_syncs
                    if left_pos < (call.lineno, call.col_offset) < right_pos
                ]
                if len(boundary_fences) != 1 or len(boundary_barriers) != 1:
                    errors.append(
                        "{} -> {} must contain exactly one CTA shared fence and one "
                        "epilog barrier; found {}/{}".format(
                            left,
                            right,
                            len(boundary_fences),
                            len(boundary_barriers),
                        )
                    )
                elif (boundary_fences[0].lineno, boundary_fences[0].col_offset) >= (
                    boundary_barriers[0].lineno,
                    boundary_barriers[0].col_offset,
                ):
                    errors.append(
                        "{} -> {} handoff must order fence before epilog barrier".format(
                            left, right
                        )
                    )

            # FC2 A/SFA fragments are hoisted once per slice.  The consumer
            # state reset is adjacent to that hoist and precedes the output
            # tile loop; neither operation may repeat per output tile.
            load_a_calls = [
                call
                for call in ordered_calls
                if dotted_name(call.func) == "self.load_fc2_a_fragments"
            ]
            reset_calls = [
                call
                for call in ordered_calls
                if dotted_name(call.func) == "phase2_cons_state.reset_count"
            ]
            if len(load_a_calls) != 1:
                errors.append(
                    "load_fc2_a_fragments must run exactly once per slice; found {}".format(
                        len(load_a_calls)
                    )
                )
            if len(reset_calls) != 1:
                errors.append(
                    "phase2_cons_state.reset_count must run exactly once per slice; found {}".format(
                        len(reset_calls)
                    )
                )
            if len(load_a_calls) == 1 and len(reset_calls) == 1:
                load_pos = (load_a_calls[0].lineno, load_a_calls[0].col_offset)
                reset_pos = (reset_calls[0].lineno, reset_calls[0].col_offset)
                if not positions["q1"] < load_pos < reset_pos < positions["fc2"]:
                    errors.append(
                        "per-slice FC2 setup must be q1 -> load_fc2_a_fragments -> "
                        "phase2_cons_state.reset_count -> first fc2 tile"
                    )

            output_loops = []
            for node in iter_nodes(phase_loop):
                if not isinstance(node, ast.For):
                    continue
                if not (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "output_tile_idx"
                ):
                    continue
                paths = [dotted_name(call.func) for call in method_calls(node)]
                if "self.fc2_to_sC" in paths or "self.scatter_sC_to_gmem" in paths:
                    output_loops.append(node)
            if len(output_loops) != 1:
                errors.append(
                    "expected one output_tile_idx loop containing FC2 + Scatter; found {}".format(
                        len(output_loops)
                    )
                )
            else:
                output_loop = output_loops[0]
                output_calls = sorted(
                    method_calls(output_loop),
                    key=lambda call: (
                        getattr(call, "lineno", 0),
                        getattr(call, "col_offset", 0),
                    ),
                )
                output_event_paths = {
                    "self.fc2_to_sC": "fc2",
                    "cute.arch.fence_proxy": "fence",
                    "self.epilog_sync_barrier.arrive_and_wait": "epilog",
                    "self.scatter_sC_to_gmem": "scatter",
                }
                output_events = [
                    output_event_paths[dotted_name(call.func)]
                    for call in output_calls
                    if dotted_name(call.func) in output_event_paths
                ]
                expected_output_events = [
                    "fc2",
                    "fence",
                    "epilog",
                    "scatter",
                    "epilog",
                ]
                if output_events != expected_output_events:
                    errors.append(
                        "each output tile must be fc2 -> fence -> epilog -> scatter -> "
                        "post-scatter epilog; observed {}".format(
                            " -> ".join(output_events) or "none"
                        )
                    )
                output_fences = [
                    call
                    for call in output_calls
                    if dotted_name(call.func) == "cute.arch.fence_proxy"
                ]
                if len(output_fences) != 1 or not fence_is_cta_shared(output_fences[0]):
                    errors.append(
                        "output-tile FC2/Scatter handoff requires exactly one "
                        "fence_proxy('async.shared', space='cta')"
                    )

                output_nodes = set(iter_nodes(output_loop))
                if load_a_calls and load_a_calls[0] in output_nodes:
                    errors.append(
                        "load_fc2_a_fragments must remain outside output tile loop"
                    )
                if reset_calls and reset_calls[0] in output_nodes:
                    errors.append(
                        "phase2_cons_state.reset_count must remain outside output tile loop"
                    )
                pass_final_call = phase_events[-1][1]
                if pass_final_call in output_nodes:
                    errors.append("pass_final arrive must occur after output tile loop")

    fc2_resets = [
        call
        for call in method_calls(fc2)
        if dotted_name(call.func) == "phase2_cons_state.reset_count"
    ]
    if fc2_resets:
        errors.append("fc2_to_sC must not reset phase2_cons_state per output tile")

    # Splitting FC2 and Scatter is equivalent only while one epilogue M tile
    # spans the full CTA M tile.  Prove the shared constructor origin and lock
    # both helper loops to the same ratio; comments or runtime assumptions are
    # not accepted as evidence.
    initializer = methods.get("__init__")
    if initializer is None:
        errors.append("MoEDynamicKernel must define __init__ for epilogue shape lock")
    else:
        tile_shapes = assigned_values(initializer, "self.tile_shape_mnk")
        epi_tiles = assigned_values(initializer, "self.epi_tile")
        if len(tile_shapes) != 1 or len(epi_tiles) != 1:
            errors.append(
                "__init__ must assign self.tile_shape_mnk and self.epi_tile exactly once"
            )
        else:
            tile_shape = tile_shapes[0]
            epi_tile = epi_tiles[0]
            valid_shape_tuples = (
                isinstance(tile_shape, (ast.Tuple, ast.List))
                and len(tile_shape.elts) >= 1
                and isinstance(epi_tile, (ast.Tuple, ast.List))
                and len(epi_tile.elts) >= 1
            )
            if not valid_shape_tuples:
                errors.append(
                    "tile_shape_mnk and epi_tile must expose literal tuple/list M origins"
                )
            elif not (
                is_index(tile_shape.elts[0], "mma_tiler_mn", 0)
                and is_index(epi_tile.elts[0], "mma_tiler_mn", 0)
            ):
                errors.append(
                    "tile_shape_mnk.M and epi_tile.M must both originate from "
                    "mma_tiler_mn[0]"
                )

    for helper_name in ("fc2_to_sC", "scatter_sC_to_gmem"):
        helper = methods[helper_name]
        epi_assignments = []
        for node in iter_nodes(helper):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "epi_rest_m":
                epi_assignments.append(node.value)
        if len(epi_assignments) != 1 or not is_epi_rest_expression(epi_assignments[0]):
            errors.append(
                "{} must assign exactly `epi_rest_m = "
                "self.tile_shape_mnk[0] // self.epi_tile[0]`".format(helper_name)
            )

        epi_loops = []
        for node in iter_nodes(helper):
            if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
                continue
            if dotted_name(node.iter.func) != "cutlass.range_constexpr":
                continue
            if (
                node.iter.args
                and isinstance(node.iter.args[0], ast.Name)
                and node.iter.args[0].id == "epi_rest_m"
            ):
                epi_loops.append(node)
        if len(epi_loops) != 1:
            errors.append(
                "{} must have exactly one range_constexpr(epi_rest_m) loop; found {}".format(
                    helper_name, len(epi_loops)
                )
            )

    # pass_gate is an intentional early handoff: every syntactic FC1 release
    # precedes it, it is guarded by the final half, and SwiGLU follows it.
    fc1 = methods["fc1_gate_up_swiglu_to_sC"]
    fc1_parents = parent_map(fc1)
    fc1_calls = method_calls(fc1)
    releases = [
        call
        for call in fc1_calls
        if dotted_name(call.func) == "ml_pipeline.consumer_release"
    ]
    gate_arrivals = [
        call
        for call in fc1_calls
        if dotted_name(call.func) == "self.pass_gate_barrier.arrive_unaligned"
    ]
    swiglu_calls = [
        call for call in fc1_calls if dotted_name(call.func) == "gated_activation_f32"
    ]
    if not releases:
        errors.append("FC1 helper must visibly release ml_pipeline stages")
    if len(gate_arrivals) != 1:
        errors.append(
            "FC1 helper must contain exactly one pass_gate arrive; found {}".format(
                len(gate_arrivals)
            )
        )
    if len(swiglu_calls) != 1:
        errors.append(
            "FC1 helper must contain exactly one SwiGLU call site; found {}".format(
                len(swiglu_calls)
            )
        )
    if releases and len(gate_arrivals) == 1 and len(swiglu_calls) == 1:
        arrive = gate_arrivals[0]
        arrive_pos = (arrive.lineno, arrive.col_offset)
        final_release_pos = max((call.lineno, call.col_offset) for call in releases)
        swiglu_pos = (swiglu_calls[0].lineno, swiglu_calls[0].col_offset)
        if not final_release_pos < arrive_pos < swiglu_pos:
            errors.append(
                "pass_gate arrive must be after the final FC1 release and before SwiGLU"
            )

        node = arrive
        guarded_by_final_half = False
        while node in fc1_parents:
            node = fc1_parents[node]
            if isinstance(node, ast.If) and is_fc1_final_half_test(node.test):
                guarded_by_final_half = True
                break
            if node is fc1:
                break
        if not guarded_by_final_half:
            errors.append("pass_gate arrive must be guarded by `fc1_half == 1`")

    # The handoff belongs to FC1 only; duplicated/moved arrivals are unsafe.
    all_gate_arrivals = [
        call
        for method in methods.values()
        for call in method_calls(method)
        if dotted_name(call.func) == "self.pass_gate_barrier.arrive_unaligned"
    ]
    if len(all_gate_arrivals) != 1:
        errors.append(
            "pass_gate arrive must have exactly one call site in MoEDynamicKernel; found {}".format(
                len(all_gate_arrivals)
            )
        )

    return errors


def default_source_path():
    return Path(__file__).resolve().parents[2] / "moe_dynamic_kernel_opt.py"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=default_source_path(),
        help="candidate kernel source (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        source = args.source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print("FAIL: unable to read {}: {}".format(args.source, exc), file=sys.stderr)
        return 2

    errors = validate_source(source, str(args.source))
    if errors:
        print("FAIL: phase structural gate rejected {}".format(args.source))
        for error in errors:
            print("  - {}".format(error))
        return 1

    print("PASS: phase structural gate accepted {}".format(args.source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
