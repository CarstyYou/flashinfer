# CUTLASS Fused MoE Multi-Kernel Pipeline — SM120 NVFP4 Baseline

本文记录当前 benchmark 中 `flashinfer.fused_moe.cutlass_fused_moe` 的源码执行模型，用作 [单-launch CuTeDSL 设计](kernel_design.md) 的 baseline。它是 host 端编排的多-kernel pipeline，不是一个覆盖完整 MoE 的 CUDA kernel。

Source identity: `flashinfer @ 517cca9c2e7d91f524fcb5f078370c056308d461`

证据状态：

- **source-confirmed**：当前源码直接确定。
- **tactic-dependent**：由 GEMM tactic、cache 或运行参数决定。
- **profile-pending**：源码推导的运行时预期，尚未由 NSys 证实。

## Fixed bench contract

- GPU path：SM120；`M=1..8192`，`H=2048`，`I_tp=512`，`E=256`，`top_k=8`。
- FC1/FC2 权重与输入 activation 均为 NVFP4，block scale size 为 16；输出为 BF16；activation 为 SwiGLU。
- `topk_ids` 和 `topk_weights` 已由上游给出；CUTLASS pipeline 不计算 router logits、softmax 或 top-k。
- `min_latency_mode=False`、无 LoRA、无 W4 groupwise、默认 `use_fused_finalize=True`。
- `fp4_quantize(x)` 在 benchmark 的 timed closure 外完成；下文不把 BF16→NVFP4 输入量化计入 pipeline。

## Multi-kernel roadmap

```text
topk_ids / topk_weights + prequantized NVFP4 input
  → route sort + permutation maps                    1 or 3 kernels
  → expand/permute input rows and scale factors      1 kernel
  → grouped-GEMM TMA metadata/stride setup           1 kernel
  → CUTLASS grouped GEMM1                            1 kernel
  → SwiGLU + FC2-input NVFP4 requant + scale         1 kernel
  → CUTLASS grouped GEMM2                            1 kernel
  → finalize: GEMM2 epilogue fusion or standalone    0 or 1 kernel
  → BF16 output
```

PDL 只建立相邻 launch 的依赖，不把这些 kernels 合并成一个 kernel。

## Source-confirmed control flow

### Route prologue: 1 or 3 kernels

- `M <= 256`、`top_k ∈ {1,2,4,6,8}`、expert encoding 不超过 9 bits 且 shared memory 足够时，尝试一个 `fusedBuildExpertMapsSortFirstTokenKernel`。当前 `E=256, top_k=8` 满足静态条件。
- 否则固定回退为三个 kernels：`blockExpertPrefixSumKernel → globalExpertPrefixSumKernel → mergeExpertPrefixSumKernel`。
- 因此 `M > 256` 必走三-kernel route；`M <= 256` 是单-kernel eligible，不代表 runtime 已证实成功。

### GEMM1 → activation/requant → GEMM2

- SM120 NVFP4 TMA path 的 GEMM1 是一次 grouped GEMM launch。
- 该路径显式禁止 Ampere gated-activation fusion；随后单独 launch `doActivationKernel`，完成 SwiGLU、FC2 输入 NVFP4 requant 和 scale 写出。
- GEMM2 是另一次 grouped GEMM launch，消费量化后的 FC2 activation。

### Finalize branch

- `use_fused_finalize=True` 只是让 GEMM2 同时枚举 FINALIZE tactics；实际是否融合仍由选中的 GEMM2 tactic 决定。
- FINALIZE tactic：top-k reduction/unpermute 融入 GEMM2 epilogue，不再 launch finalize kernel；GEMM2 前另有一次 `cudaMemsetAsync(output)`。
- non-FINALIZE tactic：GEMM2 写 expanded expert rows，再 launch `finalizeMoeRoutingKernel` 完成 unpermute、router scale 和 top-k reduction。

## Tactic-dependent behavior

- GEMM1 与 GEMM2 分别选 tactic；tuning/cache 可以改变 tile、schedule 和 GEMM2 finalize 分支。
- 非 tuning 模式 cache miss 返回 fallback `-1`。runner 对 GEMM1/GEMM2 各取其子列表首项；FINALIZE configs 在基础 configs 之后追加，因此干净进程、无已加载 cache 时预期选择 non-FINALIZE GEMM2。
- 上述 fallback 是源码规则；当前 benchmark 的实际 tactic 和 timeline 仍属于 **profile-pending**。

## Expected steady-state launch count

下表是 **profile-pending** 源码模型。假设 SM120 NVFP4 选择 TMA grouped-GEMM tactics；排除 timed closure 外的输入量化、首次 JIT/autotune、L2 flush 和 CUDA Graph 管理操作。

| Route branch | GEMM2 finalize | Expected CUDA kernels | Extra CUDA operation |
|---|---|---:|---|
| fused route，`M <= 256` 且成功 | standalone | 7 | — |
| fused route，`M <= 256` 且成功 | fused epilogue | 6 | `cudaMemsetAsync(output)` |
| 3-step route，`M > 256` 或 fused route 回退 | standalone | 9 | — |
| 3-step route，`M > 256` 或 fused route 回退 | fused epilogue | 8 | `cudaMemsetAsync(output)` |

`M <= 256` 若 fused route 在 runtime 回退，使用三步 route 行，即增加 2 个 kernels。
按上述无 cache fallback，当前 benchmark 源码预期落在 standalone 两行；仍待 NSys 确认。

## Source map

| Area | Source |
|---|---|
| Bench contract and timing boundary | `../w4a4_moe_bench/scripts/bench_qwen35_w4a4_moe_backends.py:44-48,531-581,628-637` |
| Python API, separate GEMM tuning, `run_moe` | `flashinfer/fused_moe/core.py:524-644,856-1099` |
| SM120 NVFP4 JIT flags | `flashinfer/jit/fused_moe.py:58-73` |
| Route 1/3 branches and main sequence | `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh:350-583,868-930,3856-3968` |
| Expand, TMA setup, GEMM1 and activation | `cutlass_fused_moe_kernels.cuh:1607-1680,2319-2430,3048-3274,4028-4194` |
| GEMM2 and finalize branch | `cutlass_fused_moe_kernels.cuh:3278-3403` |
| Tactic fallback and FINALIZE config ordering | `flashinfer_cutlass_fused_moe_binding.cu:231-242,821-855`; `moe_gemm_template_dispatch.h:545-669`; `flashinfer/autotuner/autotuner.py:1366-1433` |

## NSys TODO

- 对 `M=256` 与 `M=512` 各 capture 一个 steady-state call，核对 route 分界、kernel 名称、顺序和数量。
- 记录实际 GEMM1/GEMM2 tactics，确认 GEMM2 是 standalone 还是 fused finalize，并检查 fused path 的 `cudaMemsetAsync`。
- 明确排除首次 JIT/tactic preparation 和 benchmark L2 flush；确认 CUDA Graph capture/replay 没有改变被计数的 pipeline。
- profile 后把 launch-count 表从 **profile-pending** 更新为实测证据，并链接原始 `.nsys-rep` 或分析报告。
