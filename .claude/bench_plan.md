# bench_plan.md v3 — PR #3562 cute SM120 MXFP8 MoE GEMM perf 对比 (PM 要求, fixed scope)

> Source-of-truth plan, not committed (`.claude/` untracked)。v3 收敛到 PM-approvable fixed scope (从 v2 的 1680-cell routing matrix → 112-cell fixed)。

## §1 目标

PM 要求: 对 `flashinfer.grouped_mm.moe_gemm_mxfp8_nt_groupwise` (cute SM120 backend) 提供跟其他相似 grouped MoE GEMM 的性能对比, 产出 manager-facing perf report。

v2 → v3 收敛动机: unfixed (routing-driven, 1680 cell) PR 塞不下, 先 fixed (uniform m_pe, 112 cell) 进 PR; routing 版作 future work, 分 script。

## §2 对比 op 全集 (4 op)

| # | API | Backend | Repo | Mode / Entry (locked) | Scale dtype |
|---|---|---|---|---|---|
| 1 | `moe_gemm_mxfp8_nt_groupwise` | cute SM120 | flashinfer (我们 / subject) | **WithZeroPadding** (API 唯一 expose) | UE8M0 |
| 2 | `grouped_mm_mxfp8` | cudnn | flashinfer | grouped (1 entry) | UE8M0 |
| 3 | `m_grouped_fp8_gemm_nt_contiguous` | DeepGEMM | mega_inference/DeepGEMM (leavelet/sm120 HEAD `76e93aa`) | **contiguous** (locked, 非 masked) | FP32 或 UE8M0 (recipe-决定) |
| 4 | `group_gemm_fp8_nt_groupwise` | cutlass | flashinfer | grouped (`backend="cutlass"`, `scale_granularity_mnk` 参数化) | FP32 |

## §3 Scale layout 支持矩阵 — granK 决定哪些 op 入场

| Op | granK=32 | granK=128 | 备注 |
|---|---|---|---|
| #1 cute SM120 (we) | ✅ | ✅ | dispatch 双 path 都 instantiated |
| #2 cudnn | ✅ | ❌ | industry MX fixed at 1×32 |
| #3 DeepGEMM leavelet | ✅ | ✅ | recipe 参数化 |
| #4 cutlass FP8 | ✅ (1, 1, 32) | ✅ (1, 1, 128) | scale dtype FP32; 1D weight scale 跟 #1/#2/#3 一致 |

→ **granK=32 测 op #1/#2/#3/#4** (4 op 全集)
→ **granK=128 测 op #1/#3/#4** (3 op; 排除 cudnn)

## §4 Bench matrix (fixed scope)

### Fixed config

| 维度 | 值 |
|---|---|
| Op shape | grouped GEMM (8 groups) |
| Layer fc1 weight (N, K) | (4096, 7168) |
| Layer fc2 weight (N, K) | (7168, 4096) |
| num_expert | 8 (= num_groups) |
| m_pe (per-expert M, uniform) | {1, 4, 8, 16, 192, 256, 1024, 4096} |
| Total M (per cell) | 8 × m_pe |

### Cell count

- 8 m_pe × 2 layer (fc1, fc2) = **16 cell per (op, granK)**
- (op, granK) pair: granK=32 全 4 op + granK=128 排 cudnn = **7 pair**
- **Total = 16 × 7 = 112 cell** per hardware phase

### Drop (vs v2)

- ❌ 3 customer profiles (TP2UP2 / DSv4_TP8 / FI_TEST)
- ❌ 4 skew scenarios (S0_uniform / S1_light / S2_heavy / S3_bimodal) — uniform m_pe 不需要 skew 维度
- ❌ Batch sweep + routing (B → topk → per-expert m 推导) — 直接给 m_pe, 跳过 routing

### Caller-side padding contract per backend

每 backend 客户必须 pad to 该 op API contract; PR narrative 的核心 = cute SM120 ZeroPadding kernel 内部低成本 pad, 不让客户 memory pad → small m_pe 区间 dominantly win。

| Backend | Per-expert M alignment (caller-side) | m_pe=1 实际 m_pe_padded | 备注 |
|---|---|---|---|
| cute SM120 | 无 (kernel ZeroPadding 内部处理) | 1 | 主角 |
| cudnn MXFP8 | 无 (API doc 未列 m_indptr alignment) | 1 | P1 verify |
| DeepGEMM contiguous | `deep_gemm.get_mk_alignment_for_contiguous_layout()` (典型 128) | 128 | 浪费 127 行 |
| cutlass FP8 groupwise | 4 (m_indptr 每值 % 4 == 0) | 4 | 浪费 3 行 |

### 度量

- 唯一 perf 度量 = `t_us` (50-rep median, l2-flush per iter)
- 不计算 TFLOPS (PM 只看耗时)
- CSV 加 `m_pe_padded` 列让 report 能讲 pad overhead

## §5 Hardware + env

两阶段 hardware:

| 阶段 | GPU | 节点 | 备注 |
|---|---|---|---|
| **Phase A (now)** | **6K Pro Server-Edition** (NVIDIA RTX PRO 6000 Blackwell Server Edition) | Slurm `2574615`, Docker `186e56e22fce` | 第一版 perf 全集在此跑 |
| **Phase B (later)** | **5K Pro** (NVIDIA RTX PRO 5000 Blackwell) | xiy@10.6.142.16, enroot container `fi-ci-cu130` | 复跑同 plan, 验证 cross-GPU trend |

Software (两阶段共用):

| 项 | 值 |
|---|---|
| flashinfer | editable install (现 `xiy/sm120_group_gemm_mxfp8` HEAD `db7abc0`) |
| DeepGEMM 编译 | `cd mega_inference/DeepGEMM && pip install -e .` (leavelet/sm120 HEAD `76e93aa`) |
| cudnn | 已随 flashinfer 安装, 检查 `cudnn.backend_version()` ≥ 9.x |
| cute SM120 kernel | 已 ship 在 flashinfer HEAD; granK ∈ {32, 128} dispatch 已 built-in |

## §6 Bench loop (per cell, mirror 6KD `bench_real_dist.py`)

```python
# Per cell: pick (m_pe, layer, op, granK)
n, k = layer_shape[layer]          # fc1: (4096, 7168), fc2: (7168, 4096)
per_expert_m = [m_pe] * num_expert # uniform
a, b, a_scale, b_scale, m_indptr = make_inputs(per_expert_m, n, k, granK)

# Warmup 10 + measure 50, l2-flush per iter
flush_buf = torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda")
for _ in range(10):
    op(a, b, a_scale, b_scale, m_indptr, ...)
torch.cuda.synchronize()

times_us = []
for _ in range(50):
    flush_buf.zero_()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record(); op(...); end.record()
    torch.cuda.synchronize()
    times_us.append(start.elapsed_time(end) * 1e3)
t_us = sorted(times_us)[len(times_us)//2]  # median
```

## §7 Output

Per-cell CSV row:

```
layer (fc1|fc2), n, k, num_expert, m_pe, m_pe_padded, total_rows,
op, granK, t_us
```

`m_pe` = logical (customer 送 input); `m_pe_padded` = caller-pad 后实际送 kernel; `t_us` = 50-rep median latency。

- 路径: `flashinfer/.claude/6kpro_bench_results/<date>_pr3562_perf/<op>_<granK>.csv`
- Manager-facing report: `report.md` 用 `kernel-perf-report` skill 5 章 template (TL;DR / 关键数据 / 主要实现差异 [optional] / 结论 & 观察 / 细节 link); cute branding

## §8 Correctness (skip)

纯 perf sweep, 不跑 reference correctness。信任:
- PR #3562 cute SM120 op 已在 `tests/gemm/test_cute_sm120_mxfp8.py` 64/64 PASS verified
- DG / cudnn / cutlass 各自上游 reference 实现已 trust

## §9 Phasing

| Phase | 内容 | 产出 |
|---|---|---|
| P1 — Infra setup | (a) 编译 DeepGEMM leavelet/sm120 + import sanity (b) 4 op 在 m_pe=16 small shape 跑通 | "4 op runnable" gate |
| P2 — Bench sweep | 跑 §4 matrix (16 cell × 7 op-granK pair = 112 cell), 输出 §7 CSV | 6kpro_bench_results/*.csv |
| P3 — Report | 5 章 manager-facing report | report.md |

## §10 File layout (3 script, untracked under `flashinfer/.claude/bench/`)

| 文件 | 职责 | 优先级 |
|---|---|---|
| `_bench_common.py` | 共享 utils: 4 backend dispatch / l2-flush / 50-rep median / csv writer / input maker | P0 (write first) |
| `bench_grouped_gemm_fixed.py` | PM PR scope: 112 cell sweep main (本 plan) | P0 |
| `bench_moe_gemm_routing.py` | Future unfixed: routing + 3 customer × 4 scenario × 10 batch (v2 plan revival) | P1 (后续) |

## §11 Open questions (decision 给 xiy)

1. **Scale dtype mismatch**: op #3/#4 是 FP32 scale, op #1/#2 是 UE8M0 — 同 granK 对比时 dequant 路径有差异, 是否要在 report §3 "主要实现差异" 单列一句?
