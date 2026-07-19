# exp_008：Branch-paired 双 N64 A/SFA 复用

Status: **Completed / Accepted**

## 1. 目标

在 exp_007 temporal 双 N64 zero-spill candidate 上消除非必要的 A/SFA replay：每个 N64 half 只加载一次
A/SFA，同时加载 Gate/Up 两组 B/SFB，并在同一 K-loop 内交错执行两支 OMMA，随后立即 SwiGLU。

实验依次回答：

1. 新实现是否保持正确性、Tensor work 和 zero-spill；
2. 未插桩 E2E latency 是否优于 exp_007 candidate，并相对 8-warp N128 anchor 如何；
3. 完成未插桩测量后，插桩 phase breakdown 能否解释性能变化。

production kernel、exp_007 overlays 与历史实验保持不变，只迭代
`.claude/w4a4_moe_bench/moe_dynamic_kernel_opt.py` 并生成新的 immutable overlay。

## 2. 对比臂与唯一改动

| Arm | 作用 | Source contract |
|---|---|---|
| `anchor_8warp_n128` | net baseline | exp_007 immutable N128 overlay，Gate N128 → Up N128 |
| `temporal_n64_v0` | primary baseline | exp_007 immutable candidate，两个 half；每支 Gate/Up 各自加载 A/SFA |
| `branch_paired_n64_v1` | candidate | 每个 half 共用一次 A/SFA；Gate/Up OMMA 在同一 K-loop 内配对 |

Candidate 每个 logical N128 slice 的目标顺序：

```text
half 0:
  TMA [A, SFA, B_gate64, SFB_gate, B_up64, SFB_up]
  for K: OMMA Gate64; OMMA Up64
  SwiGLU → sC[:, 0:64]
half 1:
  TMA [A, SFA, B_gate64, SFB_gate, B_up64, SFB_up]
  for K: OMMA Gate64; OMMA Up64
  SwiGLU → sC[:, 64:128]
Q1 N128 once → FC2 once → scatter once
```

实现锁定为：现有 `sB/sSFB` backing 用于 Gate（随后由 FC2 复用），新增独立的 `sB_up` N64 backing 与
`sSFB_up` physical-N128 backing；不尝试让两支 physical-N128 SFB 非法重叠。逐 stage machine-check range、
alignment 和不重叠性。单 FC1 pipeline 的 `tx_count` 精确等于
`A + SFA + B_gate64 + SFB_gate128 + B_up64 + SFB_up128`；每 stage 一次 wait，在两支 OMMA 都消费后才
release。双 half 完成后沿用 CTA fence/barrier，再把 aliased Gate/FC2 backing 交给 FC2。

禁止改变 warp 数、tile、stage 数、scheduler、route/Q0、Q1、FC2、scatter、数学模式和输入；不得通过降低
stage/tile/work 绕过容量问题。新增 storage 必须报告最终 SMEM 与 occupancy。

## 3. 必须成立的工作契约

- 每个 half 的 A/SFA producer trip count 从 Gate+Up 两次降为一次；全 slice A/SFA bytes 必须从 exp_007 v0 的
  2× anchor 恢复到 1× anchor。
- B bytes、static OMMA opcode/count、matched executed Tensor instructions 与 FP4 Tensor ops 三臂一致。
- Gate/Up B 与 SFB 地址覆盖 half0/half1、四个 logical slice，不能错支、漏算或重算。
- 对 logical slice `s∈[0,4)`、half `h∈[0,2)`：Up B index=`2s+h`，Gate B index=`2(s+4)+h`；
  Up/Gate physical SFB index 分别为 `s` / `s+4`，consumer 再选择 half `h`。ledger 与 canary 必须逐项验证。
- 每个 half 同时只保留一对 N64 Gate/Up accumulator；SwiGLU 后生命周期不得跨入下一 half。
- Q1、FC2、scatter 各一次；scheduler、FC2/scatter executable source 与 exp_007 v0 保持 identical。
- Physical SFB 若仍受 N128 helper 限制，可以保留 replay，但必须单独计数，不能混入“A/SFA 已复用”的结论。

## 4. Gates 与执行顺序

### Gate A：编译、结构、正确性、静态资源

1. Fresh JIT，锁 source/PTX/cubin/SASS hash、kernel symbol、grid `(1,1,110)`、block `(288,1,1)`。
2. Machine-check producer trip/bytes ledger、Gate/Up/half 地址和一次 Q1/FC2/scatter。
3. 跑 M256、M8192、`sparse_empty`、`exact_128`、`tail_129`、`hot_expert`、Gate/Up branch-half-slice
   canary；沿用 exp_007 固定 correctness thresholds，不得看到结果后放宽。
4. 每个 distinct candidate cubin 都必须 `STACK=0`、compiler SpillRefill/local SASS=0；否则 reject，不进入性能比较。

### Gate B：先测未插桩 E2E

- 环境：同一 dedicated 5KP UUID，graphics clock 2377 MHz，无 foreign process；固定 26.05 image、CuteDSL
  4.6.0、CUDA/nvcc/ptxas 13.2.78、FI/CUTLASS/source/dependency hash。
- Cases：canonical M256、M8192，`E=256,H=2048,I_tp=512,topk=8`。
- Boundary：完整 fused MoE CUDA Graph node；计时不含 JIT、graph capture、host launch 与 L2 flush。
- Protocol：每次 192 MiB L2 flush；warmup=5、timed=50；A-B-B-A × 5 groups；每 position 独立进程。
- Primary pair：`temporal_n64_v0` vs `branch_paired_n64_v1`；secondary pair：`anchor_8warp_n128` vs v1。
- 汇总 median/p10/p90/CV、paired speedup 与 10k bootstrap 95% CI；speedup=`baseline/v1-1`。
- 统计单位是独立 process 的 ABBA position；bootstrap 以 ABBA group 为 paired unit，不把 50 个 replay 当作
  50 个独立样本。等价带预注册为 `[-2%, +2%]`。
- 任一 source/cubin/GPU/clock/JIT/correctness drift 使该 pair 无性能结论。

### Gate C：E2E 完成后才采动态与插桩证据

1. 对 v1 M8192 未插桩 cubin 采 matched NCU 四项 dynamic spill metrics、Tensor instructions 与 FP4 Tensor ops；
   missing/`n/a` fail closed，spill 四项必须为 0，Tensor work 必须与 baselines 相同。若 M256 产生不同 cubin，
   也采其 dynamic spill；相同 cubin则由 hash 证明复用。
2. 从 v0/v1 两个 immutable overlay 分别生成同源 marker-disabled control 与 marker-enabled probe；各用独立 JIT
   namespace。Control 必须保持 Gate-B 的 source semantics、correctness 与 work identity，但其 cubin/hash/resource
   作为独立 instrumented binary 记录，不能冒充 Gate-B cubin。
3. 先测 control/probe E2E perturbation，再只用 probe 报 phase share；插桩后的绝对 latency不得替代 Gate B。
4. Additive 公共边界为 front-end/route+Q0、claim/cache、
   `FC1 + interleaved activation envelope`、Q1、combined FC2+scatter、residual。v0/v1 都是
   `half0 FC1 → SwiGLU0 → half1 FC1 → SwiGLU1 → collective → Q1`，因此在不改变同步语义时，
   不存在能把纯 `FC1-total` 与 `activation+Q1` 同时分成 collective-wall additive 两栏的自然边界。
   `pair0 FC1` / `SwiGLU0` / `pair1 FC1` / `SwiGLU1` 由 W0–W7 same-warp delta 单独记录，
   只作 non-additive diagnostic track；不允许为测量新增 barrier，也不允许用该 track 计算 full-kernel share。
5. 关键 consumer boundary 由 W0–W7 各 warp lane0 记录并 collective rollup；不能沿用历史 W0 representative。
   首版不采 W8 producer，也不复刻 exp_006 的 FC2 16-tile细分。
6. v0/v1 只比较语义一致的公共 phase，并报告 paired phase delta。历史 exp_004/006 的 4-warp/block160 数据只作
   component reference，不直接计算 speedup。
7. Marker build 的 correctness、cubin identity、REG/SMEM/STACK/spill 必须单独报告；若插桩改变 spill，phase share
   只能作为定性诊断，不能解释未插桩 latency 的精确差值。

## 5. 判定

```text
compile / correctness / work identity / static zero-spill fail
  → reject v1

任一 M 的 95% CI 完全低于 -2%
  → reject 通用 v1；插桩只用于解释退化

两个 M 均未退化，且至少一个 95% CI 完全高于 +2%，dynamic zero-spill
  → 保留 v1；结合 phase breakdown 判断下一优化点

两个 M 方向冲突
  → reject 通用 v1；只有明确允许 shape dispatch 时才保留 shape-specific 版本

两者均落入等价带或 CI 跨越判定边界
  → 不宣称性能收益；v1 保持实验状态
```

`zero-spill ⇒ 更快`、`A/SFA replay 减半 ⇒ 等量 latency 收益` 都不是预设结论。最终只接受经过 correctness、
identity、未插桩 E2E 和 spill 证据共同支持的结论。

## 6. 产物

全部放在 `results/`：immutable overlays/diff、work ledger、correctness、static/dynamic spill、未插桩 ABBA raw+
summary、插桩 control/probe identity+phase summary、manifest、`result.md`。正式报告前执行 ex-post data audit。

## 7. 补充：Production 同机 ABBA

为把跨 GPU 的历史桥接估算替换为直接证据，在 exp_008 原 GPU 上增加
`Production → current → current → Production` 对比。Production 锁定为 source SHA-256
`94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106`，current 锁定为已接受的
`branch_paired_n64_v1` SHA-256
`f3c246817679d962a3f7160dbe8b9e68262c919e26e306f349200961fc4ac971`；不修改任何 kernel。

- Cases：M256、M8192；canonical fixture、weights、outer CUDA Graph boundary 保持不变。
- 协议：同一 GPU UUID、2377 MHz application clock、同一容器/依赖；每个 M 做 5 组 ABBA，每个位置是独立
  Python process，warmup=5、timed=50、每次 replay 前做 192 MiB L2 flush。
- 证据：两臂 source/JIT/cubin/preparation identity 必须闭合；任何 GPU、clock、fixture、sample-set drift 都
  fail closed。Production/current 跨臂还必须校验 fixture、weights、reference、image/deps、checkout/CUTLASS、
  imports 与 compiler toolchain；只补 E2E，不重复 NCU、spill 或 phase capture。
- 每组 ABBA 必须连续完成；发现部分已存在的 group 一律停止，禁止跨时段拼成完整组。Point speedup 与 bootstrap
  CI 都使用 paired-group aggregate estimator，arm median 只作描述。
- 报告：以同机 paired bootstrap speedup 取代 Production 历史估算；旧估算只保留为 provenance，不再作为结论。

## Plan Review

**Date**: 2026-07-19
**Reviewer**: isolated subagent

**Verdict**: ⚠️ Gaps — 已一次性修正；不再复审

**Gaps + suggested fix**:

- Physical-N128 SFB 无法双支重叠 alias：锁定独立 Up B/SFB backing、逐-stage range/alignment、精确六路 `tx_count`
  和一次 wait/release/handoff 契约。
- Gate/Up/half 地址约束不够可执行：补充 B/SFB index 公式并由 ledger/canary machine-check。
- Spill/work gate 对 distinct cubin 覆盖不足：每个 distinct cubin 做静态检查，matched NCU 同采 spill 与 Tensor work；
  M256 若 cubin 不同则补采动态 spill。
- 仅有 v1 probe 无法解释 v0→v1：v0/v1 都生成同源 control/probe，公共 phase 做 paired delta；两次 SwiGLU
  与 pair 边界单独记录，不增加同步。
- E2E 缺统计单位、等价阈值与方向冲突规则：锁定 ABBA group paired bootstrap、±2% 等价带和逐 case 决策树。
