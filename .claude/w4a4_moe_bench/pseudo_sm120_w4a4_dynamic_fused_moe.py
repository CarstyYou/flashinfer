"""
Progressive pseudocode for the current SM120/SM121 dynamic W4A4 fused MoE.
Source identity: flashinfer @ 517cca9c2e7d91f524fcb5f078370c056308d461
Primary source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py
Dispatch source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py
Kernel/dispatch paths are clean at this HEAD; target shape comes from the benchmark worktree.
Locked scope: BF16 input/output, NVFP4 weights/activations, SiLU gated MoE,
H=2048, I_tp=512, E=256, top_k=8, tile MNK=128x128x128. Top-k ids and
weights are inputs; this kernel does not calculate router logits or select top-k.
Detailed current-kernel pseudocode follows.
Sections: launch -> kernel -> route/pack -> view init -> task consumer. Not executable.
"""

# This file intentionally uses descriptive, undefined operations as pseudocode.
# ruff: noqa: F821, B007


# Source: flashinfer/fused_moe/cute_dsl/b12x_moe.py:549-655
# Source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py:1181-1195,1527-1768,2643-2715
# Source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py:269-324,518-689
def launch_sm120_w4a4_dynamic_fused_moe(case, tensors, workspace):
    routed_rows = case.num_tokens * 8
    static_cutover = configured_static_cutover(default=640)
    resolved_backend = "dynamic" if routed_rows > static_cutover else "static"
    require(resolved_backend == "dynamic")
    tile_m, tile_n, tile_k = 128, 128, 128
    launch = specialize_nvfp4_silu_kernel(
        tile=(tile_m, tile_n, tile_k),
        ab_stages=2,
        epilogue_stages=1,
        mma_warps=4,
        tma_warps=1,
        fc1_k_tiles=16,
        intermediate_slices=4,
        fc2_output_tiles=16,
    )
    launch(
        dynamic_fused_moe_kernel,
        tensors,
        workspace,
        grid=(1, 1, min(max_active_clusters_at_occupancy_1, sm_count)),
        block=(160, 1, 1),
        cluster=(1, 1, 1),
    )


# Source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py:691-1056,1522-1755
def dynamic_fused_moe_kernel(tensors, workspace):
    prefetch_tma_descriptors_with_warp_0()
    shared = allocate_control_route_scatter_and_ab_scale_output_buffers()
    pipelines = create_two_stage_tma_pipelines(
        gate=("A", "SFA", "W_gate", "SFB_gate"),
        up=("A", "SFA", "W_up", "SFB_up"),
        down=("W_down", "SFB_down"),
    )
    route_pack_and_materialize_tasks(tensors, workspace, shared)
    task_views = initialize_fused_task_views(tensors, workspace, shared, pipelines)
    consume_fused_moe_tasks(tensors, workspace, task_views)


# Source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py:963-1520
# Source: flashinfer/cute_dsl/fp4_common.py:2385-2427
def route_pack_and_materialize_tasks(tensors, workspace, shared):
    grid_strided_zero(
        workspace.row_counts,
        workspace.expert_write_rows,
        workspace.expert_tile_base,
        workspace.task_metadata,
        workspace.queue_counters,
        tensors.bf16_output,
    )
    resident_grid_barrier_1()
    for routed_pair in grid_strided(tensors.topk_ids):
        atomic_add(workspace.row_counts[tensors.topk_ids[routed_pair]], 1)
    resident_grid_barrier_2()
    if global_thread_id == 0:
        workspace.expert_tile_base = exclusive_prefix_sum(
            ceil(workspace.row_counts[e] / 128) for e in range(256)
        )
    resident_grid_barrier_3()
    while True:
        routed_pair_batch = cta_leader_atomic_claim_and_broadcast(
            workspace.pair_head, 10
        )
        if routed_pair_batch >= len(tensors.topk_ids):
            break
        for routed_pair in warp_owned_pairs(routed_pair_batch, pairs_per_warp=2):
            expert = tensors.topk_ids[routed_pair]
            token = routed_pair // 8
            route_weight = tensors.topk_weights[routed_pair]
            expert_row = atomic_add(workspace.expert_write_rows[expert], 1)
            physical_tile = workspace.expert_tile_base[expert] + expert_row // 128
            physical_row = physical_tile * 128 + expert_row % 128
            workspace.token_map[physical_row] = token
            workspace.token_weights[physical_row] = route_weight
            for hidden_block_16 in partition_across_warp(tensors.bf16_x[token], 16):
                input_fp4x16, input_e4m3_scale = quantize_nvfp4_block(
                    hidden_block_16, global_scale=tensors.w1_alpha[expert]
                )
                workspace.packed_expert_input[physical_row, hidden_block_16] = (
                    input_fp4x16
                )
                workspace.input_sfa_mma_layout[physical_row, hidden_block_16] = (
                    input_e4m3_scale
                )
    global_fence_then_cta_barrier()
    resident_grid_barrier_4()
    for expert in experts_owned_by_cta_leader():
        for expert_m_tile, valid_rows in physical_tiles_for_expert(expert, tile_m=128):
            global_m_tile = workspace.expert_tile_base[expert] + expert_m_tile
            for intermediate_slice in range(4):
                slot = global_m_tile * 4 + intermediate_slice
                workspace.task_metadata[slot] = (
                    expert,
                    global_m_tile,
                    intermediate_slice,
                    1,
                    valid_rows,
                )
    if global_thread_id == 0:
        workspace.task_tail = workspace.expert_tile_base[256] * 4
    resident_grid_barrier_5()
    if global_thread_id == 0:
        release_store(workspace.all_work_published, 1)


# Source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py:1522-1749
def initialize_fused_task_views(tensors, workspace, shared, pipelines):
    gmem_tiles = partition_tma_views(
        workspace.packed_expert_input,
        workspace.input_sfa_mma_layout,
        tensors.fp4_w13,
        tensors.w13_sfb_mma_layout,
        tensors.fp4_w2,
        tensors.w2_sfb_mma_layout,
    )
    smem_tiles = partition_shared_A_B_SFA_SFB_and_bf16_C(shared)
    register_fragments = partition_qmma_A_B_SFA_SFB_and_fp32_accumulators(smem_tiles)
    pipeline_states = initialize_gate_up_down_producer_consumer_states(pipelines)
    set_register_budget(mma_warps=4, mma_registers=232, tma_warp=1, tma_registers=32)
    return bundle(gmem_tiles, smem_tiles, register_fragments, pipeline_states)


# Source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py:1751-2710
# Source: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_activation.py:16-52
# Source: flashinfer/cute_dsl/fp4_common.py:1798-1827,2385-2427
def consume_fused_moe_tasks(tensors, workspace, views):
    while True:
        control = cta_leader_fetch_add_head_then_publish_task_or_done_to_shared(
            acquire_load(workspace.task_tail),
            workspace.task_head,
            workspace.task_metadata,
        )
        cta_sync_and_load_shared_control(control)
        if control.done:
            break
        expert, global_m_tile, slice_begin, slice_count, valid_rows = control.task
        shared_token, shared_route_weight = cta_cache_scatter_metadata(
            workspace.token_map, workspace.token_weights, global_m_tile, valid_rows
        )
        if warp_id in range(4):
            for intermediate_slice in range(slice_begin, slice_begin + slice_count):
                gate_acc_fp32 = zero_qmma_accumulators()
                for fc1_k_tile in range(16):
                    A, SFA, gate_B, gate_SFB = mma_wait_smem_to_regs(
                        views.gate_pipeline
                    )
                    gate_acc_fp32 = nvfp4_blockscaled_qmma(
                        A, SFA, gate_B, gate_SFB, gate_acc_fp32
                    )
                mma_warps_arrive_pass_gate_without_waiting()
                up_acc_fp32 = zero_qmma_accumulators()
                for fc1_k_tile in range(16):
                    A, SFA, up_B, up_SFB = mma_wait_smem_to_regs(views.up_pipeline)
                    up_acc_fp32 = nvfp4_blockscaled_qmma(
                        A, SFA, up_B, up_SFB, up_acc_fp32
                    )
                gate_fp32 = tensors.w1_alpha[expert] * gate_acc_fp32
                up_fp32 = tensors.w1_alpha[expert] * up_acc_fp32
                swiglu_fp32 = gate_fp32 * sigmoid(gate_fp32) * up_fp32
                shared_bf16_activation = mma_regs_to_shared_bf16(
                    swiglu_fp32, valid_rows
                )
                fence_shared_writes_then_mma_only_barrier()
                activation_fp4, activation_sfa = shared_bf16_to_nvfp4_and_e4m3_scales(
                    shared_bf16_activation, global_scale=tensors.fc2_input_scale[expert]
                )
                fence_shared_writes_then_mma_only_barrier()
                activation_A, activation_SFA = load_shared_activation_once_to_regs(
                    activation_fp4, activation_sfa
                )
                for output_tile in range(16):
                    down_B, down_SFB = mma_wait_smem_to_regs(views.down_pipeline)
                    down_acc_fp32 = nvfp4_blockscaled_qmma(
                        activation_A,
                        activation_SFA,
                        down_B,
                        down_SFB,
                        zero_qmma_accumulators(),
                    )
                    partial_bf16 = mma_regs_to_shared_bf16(
                        tensors.w2_alpha[expert] * down_acc_fp32, valid_rows
                    )
                    fence_shared_writes_then_mma_only_barrier()
                    for row, vector_8 in warp_quadrant_vectors(
                        partial_bf16, output_tile
                    ):
                        atomic_add_bf16x8(
                            tensors.bf16_output[shared_token[row], vector_8.columns],
                            shared_route_weight[row] * vector_8,
                        )
                    mma_only_post_scatter_barrier()
                mma_warps_arrive_pass_final_without_waiting()
        elif warp_id == 4:
            for intermediate_slice in range(slice_begin, slice_begin + slice_count):
                for fc1_k_tile in range(16):
                    tma_submit_A_SFA_gate_B_SFB(
                        views.gate_pipeline,
                        global_m_tile,
                        expert,
                        intermediate_slice,
                        fc1_k_tile,
                    )
                tma_warp_waits_for_mma_pass_gate_arrivals()
                for fc1_k_tile in range(16):
                    tma_submit_A_SFA_up_B_SFB(
                        views.up_pipeline,
                        global_m_tile,
                        expert,
                        intermediate_slice,
                        fc1_k_tile,
                    )
                for output_tile in range(16):
                    tma_submit_down_B_SFB(
                        views.down_pipeline, expert, intermediate_slice, output_tile
                    )
                tma_warp_waits_for_mma_pass_final_arrivals()
    if warp_id == 4:
        tma_drain_gate_up_down_pipelines()
