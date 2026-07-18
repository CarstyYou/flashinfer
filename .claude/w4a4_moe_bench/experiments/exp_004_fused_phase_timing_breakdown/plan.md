# exp_004：Whole Fused-Kernel Phase Timing Breakdown

Status: **closed — diagnostic whole-kernel breakdown complete**

## 1. Goal

在固定 `M=8192` case 上，测量 `MoEDynamicKernel` 从 entry 到 return 的完整 phase 分布：

```text
entry/prologue → P0 Clear → P1 Histogram → P2 Prefix
→ P3 Route+Q0+Pack → P4 Publish → compute setup
→ [T0 Claim/cache → Gate → Up → SwiGLU/Q1
   → FC2 setup → FC2 GEMM → FC2 epilogue/scatter] × task
→ task/control gaps → final no-task exit → W4 producer tail → return
```

P3 内 Route、Q0、Pack 按 routed pair 交错，没有三个共同 global boundary，只报告合并的 P3。
W4 producer 与 W0–W3 consumer 重叠，作为 non-additive track 单列。

本实验只回答 phase 时间分布和后续调查优先级，不把高占比直接判成 bottleneck，不按 phase share
拆分 whole-launch NCU counters，也不修改 production kernel。

## 2. Locked Case

| Field | Value |
|---|---|
| Backend | `cutedsl_bf16_fused` |
| Shape | `M=8192, E=256, H=2048, I_tp=512, topk=8` |
| Launch | grid `(1,1,110)`；block `(160,1,1)`；1 CTA/SM |
| Warp roles | W0–W3 MMA consumers；W4 TMA producer |
| Task population | `task_tail=2536`；capacity `3068` |
| Production kernel SHA-256 | `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106` |
| Production dispatch SHA-256 | `cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4` |
| Canonical SASS SHA-256 | `34b4c38161642a27ca6b4ec41ffad0bd70f6ff99fd8118997a4b2416c5e3abba` |

## 3. Measurement Contract

Experiment-owned overlay 使用 `%globaltimer`；所有 SM 共享时间域。CTA 记录 14 个 entry/global phase/exit
事件，每个有效 task 记录 65 个 task/consumer/W4 事件。

Gate 到 FC2 的 additive boundary 由 **W0 lane0** 记录，表示同一 CTA 的代表性 consumer timeline；它不是
W0–W3 四个 warp 的指令时间求和或精确 phase union。phase interval 内的同步等待仍属于该 interval。

主分母是 SM-equivalent wall：

```text
global_wall = max(all CTA final) - min(all CTA entry)
D = grid_z × global_wall
```

各 CTA 的互斥 interval 求和。`launch skew / early-finish idle` 与
`task control / final drain / producer tail` 显式吸收其余合法区间，要求 phase sum 与 `D` 精确闭合。
后一个 residual 包括 inter-task gaps、最终 no-task claim/exit 和 W4 producer tail。

W4 Gate/Up/Down producer 与 wait interval 只用于 overlap 观察，不进入 additive 100%。

## 4. Gates

- 5/5 graph replay correctness PASS；workspace contract PASS。
- 每次 task event `164840/164840`、CTA event `1540/1540`；未使用 task slot 保持 sentinel。
- CTA/task event 单调；2536 tasks 全部映射并限制在所属 CTA span。
- 每次及 aggregate denominator exact closure，delta 必须为 0。
- CUDA Event 与 `%globaltimer` 使用同一种聚合口径交叉检查。
- matched no-marker control 与 probe 分别保留 latency、cubin/PTX/SASS、REG/SMEM/STACK/LDL/STL。
- 若 probe 改变 binary/resource/spill，最终 classification 必须是 `diagnostic-only`。

## 5. Closure

所有 coverage/correctness/event/closure gate PASS。Probe 相对 matched control 改变了 stack 和 local SASS：

| Arm | REG | STACK | LDL | STL | SASS identity |
|---|---:|---:|---:|---:|---|
| No-marker control | 255 | 488 B/thread | 122 | 68 | canonical |
| Probe | 255 | 464 B/thread | 135 | 64 | drifted |

因此结果定级为 **diagnostic whole-kernel estimate**，不能称 production-exact timing。

Canonical artifacts：

- [result.md](results/result.md)
- [whole_kernel_timing.json](results/whole_kernel_timing.json)
- [whole_kernel_capture_summary.json](results/derived/whole_kernel_capture_summary.json)
- [manifest.json](results/manifest.json)

## Historical Attempts

本实验早期的 `%clock64` / 306-tick consumer-only probe、formal zero-write failure、局部 T1–T4 estimate
均由 Git 历史和旧 raw/derived artifacts 保留，但已全部被本 plan supersede，不再是当前 measurement contract、
主分母或 canonical result。不得用旧 `run_exp004.py refresh-manifest` 覆盖当前 whole-kernel manifest。
