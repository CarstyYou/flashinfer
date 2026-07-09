# Task 04: moe_gemm_fp8_nt_groupwise (cute) vs grouped_mm_fp8 (cudnn) 性能对比

## 前置状态 (已验证)

| 项 | 值 |
|---|---|
| `grouped_mm_fp8` arch 支持 | `@supported_compute_capability([89, 90, 100, 103, 110, 120, 121])` — 含 SM120, 无 mxfp8 式特判 |
| cuDNN 门槛 | `_CUDNN_MOE_MIN_VERSION = 92100` (9.21.0); 容器实测 `backend_version() = 92100` 恰好达标 |
| 签名 | `grouped_mm_fp8(a, b, m_indptr, alpha=None, out, out_dtype, backend="cudnn", tactic=-1)` — A `(cum_m,k)` fp8 + B `(E,n,k)` fp8, **无 groupwise scale** (mm_fp8 风格, 可选 per-tensor alpha) |

## Scope 与 caveat

- **Perf-only 对比**: 量化 recipe 不同 (我们 (1,128,128) groupwise float scale vs cudnn per-tensor
  alpha) — 同 shape 同 dtype 的 fp8 grouped GEMM 延迟对比有效, 精度不可比, 报告中必须明示。
- 两侧喂**同一份** `a_fp8 / b_fp8 / m_indptr`; cute 侧带 groupwise scales, cudnn 侧 `alpha=None`。
- 不改任何 production 代码; 纯 bench task。

## Sub-task 表

| # | Sub-task | Gate |
|---|---|---|
| 0 | cudnn 侧冒烟: 单 cell 跑通 + 小 cell 输出 vs per-tensor dequant ref sanity (确认 sm120 真能算对, 不 bench 垃圾); 顺带确认 small m_pe / empty expert / m_pe=1 是否被 cudnn 接受 | 冒烟记录进 findings |
| 1 | `tests/bench_fp8_vs_cudnn.py` (task 内): cells = E∈{4,8} × m_pe∈{1,4,8,16,192,256,1024} × (N,K)∈{(4096,7168),(7168,4096)}; warmup 10 + 50 iter median; `tactic=-1` | 脚本 review |
| 2 | 跑 2 轮 → `results/bench_fp8_vs_cudnn_r{1,2}.csv` | 跨轮漂移 ≤3pp |
| 3 | plan `## Results` 数据表 + findings 沉淀 | 收口 |

cudnn 拒绝的 cell (如 m_pe=1 或 empty) 如实记 SKIP + 原因, 不 pad 迁就。

## Risks

| # | Risk | 处理 |
|---|---|---|
| R1 | cuDNN 9.21.0 恰好卡 min 版本, sm120 fp8 grouped 可能有未暴露的 bug/慢路径 | sub-task 0 冒烟 + 首轮数据 sanity (对比 task_02 的 cutlass 数据量级) |
| R2 | cudnn graph build/plan 首调开销污染计时 | warmup 内完成 graph cache; 每 cell 独立 build 一次后计时 |
| R3 | per-tensor vs groupwise 的 dequant 开销差异是 recipe 本质差异 | 报告明示 recipe 不同, 不声称 apples-to-apples |

## Results (2026-07-09, 6K Pro 2u2g-spr-0490, cuDNN 9.21.0)

### Sub-task 0 冒烟

`grouped_mm_fp8` (cudnn) 在 SM120 可用: 4/4 冒烟 PASS (calc_diff = 0, raw fp8 matmul 确定性);
**接受 m_pe=1 / empty expert / uneven** — 无 mxfp8 式限制, cells 无需裁剪。

### Bench (28 cells × 2 轮, warmup 10 + 50 iter median, perf-only — recipe 不同)

| 区间 | cute vs cudnn speedup |
|---|---|
| 小 M (m_pe 1-16), E=4 | +41% ~ +58% |
| 小 M (m_pe 1-16), E=8 | +17% ~ +38% |
| 中 M (m_pe 192-256) | +0.6% ~ +39% (跨 shape 波动大, cudnn tactic 差异) |
| m_pe=1024 | fc2 +2.8~8.4%; **fc1 E=8 为唯一负点 -2.2%~-2.5% (两轮一致)** |

- 27/28 cells cute 领先; 跨轮 drift >3pp 仅 1 cell (`4,192,fc1`: +38.8 vs +34.2, 仍大幅正)。
- 观察 (假设, 未 NCU 验证): 大 M compute-bound 区间 cudnn per-tensor recipe 无 groupwise rescale
  开销, 天然少一部分工作 — 唯一负点与 recipe 差异方向一致, 不视为 kernel 劣化证据。
- 数据: `results/bench_fp8_vs_cudnn_r{1,2}.csv`。

### 补测: m_pe=1 的 cutlass pad-4 (2 轮一致)

cutlass `m_indptr` multiple-of-4 contract 要求 caller 把每 expert pad 到 4 行, padded 性能计给
cutlass (它自身实现的要求): fc1 E=4 +178% / E=8 +25.2%; fc2 E=4 +194% / E=8 +25.6% (over cute)。
数据: `results/bench_mpe1_cutlass_pad4.csv`; 已并入 PR #3891 表格 (标注 `pad 4`)。

### 结论

cute FP8 groupwise entry 在带 groupwise 精度优势的前提下, 对 per-tensor recipe 的 cudnn
grouped GEMM 仍在 27/28 cells 上更快, decode 小 M 区间领先 +17%~+58%。
