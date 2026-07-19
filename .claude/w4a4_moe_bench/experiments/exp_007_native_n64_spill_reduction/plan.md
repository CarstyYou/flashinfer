# exp_007：原生双 N64 FC1 的 Spill 验证

Status: **Closed / Accepted** — 全部 required correctness、work identity、static zero-spill 与 matched dynamic zero-spill gate 已通过；本实验不作性能结论。

## 1. 目标与假设

本实验只回答一个问题：在当前 8-math-warp 优化 kernel 中，保持逻辑 CTA 的 N128 输出、一次完整 FC2
和一次 scatter，仅把 FC1 Gate/Up 改成两个**原生 N64 half**，是否能正确消除 residual register spill。

待验证假设：当前 8-warp anchor 同时持有 Gate N128 和 Up N128，每线程约持有 `64 + 64` 个 FP32
accumulator，且 Gate 跨完整 Up 保活；将峰值 ownership 降为 Gate N64 + Up N64（约 `32 + 32`
FP32/thread），并在每个 half 后立即做 SwiGLU、结束这对 accumulator 的 lifetime，可以清除 224 B/thread
static frame 与对应的动态 spill/refill。N64 是缩短 live range 的实现手段，不是目标本身。

本实验不评价性能，不把 zero-spill 等同于更快，也不修改 production kernel。

## 2. 对比臂与唯一改动

| Arm | Source | Contract |
|---|---|---|
| `anchor_8warp_n128` | 实验启动时对 `moe_dynamic_kernel_opt.py` 的 immutable snapshot | 8 compute warps；Gate N128 → Up N128；FC2/scatter 各一次 |
| `candidate_8warp_native_n64_v0` | `moe_dynamic_kernel_opt.py` 的实验版本 snapshot | 8 compute warps；两个原生 N64 FC1 half；FC2/scatter 仍各一次 |

当前 anchor SHA-256 为 `3cd9e6a26056d9221f59ea6749cd601c25cbef017cf6e7349efe0925180407c1`；执行前必须重新校验并保存
full source、unified diff 与 SHA-256。production `flashinfer/fused_moe/core.py` 及原始 `moe_dynamic_kernel.py`、
exp_005 overlays 均不可修改。

Candidate 的设计边界：

```text
logical CTA = M128 × N128 × K128
W0–W7 compute + W8 TMA

half 0: native Gate N64 + native Up N64 → immediately SwiGLU → sC[:, 0:64]
        → Gate/Up accumulator lifetime ends
half 1: native Gate N64 + native Up N64 → immediately SwiGLU → sC[:, 64:128]
        → Gate/Up accumulator lifetime ends
full sC N128 → Q1 once into sA → FC2 once → scatter once
```

FC1 与 FC2 必须有显式分离的 compile-time contract：

- `fc1_tiled_mma`: `(M,N,K)=(128,64,128)`；独立锁定 permutation、B fragment/copy partition 与 N64 SMEM view；
- `fc2_tiled_mma`: 保持 `(128,128,128)` 的现有 permutation、fragment、SMEM view 与 FC2/scatter；
- `logical_slice_cnt=I_tp/128=4`，`native_fc1_tile_cnt=I_tp/64=8`；对 logical slice `s`、half `h`，
  Up B 坐标为 `2*s+h`，Gate B 坐标为 `2*(s+logical_slice_cnt)+h`；FC2 仍只消费 logical slice `s`；
- `sB` 的同一块物理 storage 分别建立 FC1-N64 和 FC2-N128 alias/view；pipeline 的每条 TMA operation 与
  精确 `tx_count` 必须写入 descriptor ledger 并 machine-check；
- B payload 使用真正 N64 descriptor，两个 half 的 B bytes 合计必须等于一个 N128 pass。

SM120 现有 SFB helper 会将 N64 scale-factor tile 物理补为 N128，因此 v0 **不声称 SFB descriptor 原生 N64**：
每个 N64 half 仍使用一个 N128 physical SFB block。该 SFB replay，以及 A/SFA 因两个 temporal half 产生的
replay，都是 candidate bundle 的已知/允许后果，必须精确计数，不能隐藏为“native N64 无额外流量”。

初版 gated-only，保持 `(4,2,1)` atom layout、pipeline stage、route/pack、Q0、FC2 与 scatter 逻辑不变；只允许
为上述 FC1-N64 contract、half offset、pipeline ownership 和 immediate SwiGLU 做最小 plumbing 改动。每个
candidate 版本一旦采证即 immutable；最终用于结论的版本必须 fresh 跑完全部 gate，禁止跨版本拼证据。

## 3. 工作等价性约束

候选进入 spill 判定前必须同时满足：

1. 输出通过 canonical independent reference，且 route/task population 与 anchor 一致；
2. logical CTA、grid、task scheduling、FC2 tile 数和 atomic scatter 数不变；
3. 两个 FC1 half 合计覆盖 N `[0,128)`，无漏算、重算或越界；
4. 只存在一对可复用的 `M128×N64` Gate/Up accumulator；每个 pair 必须立即完成 SwiGLU 并写入 `sC`，
   half 0 的 accumulator def-use/live range 不得跨入 half 1；
5. descriptor ledger 必须逐项记录 operand、branch、logical slice、half、global N range、trip count、bytes；
   两个 N64 B footprint 合计等于一个 N128 B footprint；A/SFA 与 physical-N128 SFB replay 的精确倍数作为
   allowed consequence 单独报告；
6. useful Tensor work 与 anchor 完全一致：正式 cubin 的 OMMA opcode histogram 与 exact static count `448`
   相同，并提供 FC1 Gate/Up/FC2 phase ledger；matched NCU 的 executed Tensor instructions 与 FP4 Tensor ops
   也必须相等；
7. 两个 half 都落入 `sC` 后只执行一次 full-N128 Q1；Q1 写满 `sA` 后只进入一次 FC2/scatter；scheduler、
   FC2 与 scatter 的 source/AST 必须与 anchor identical。

Q1 不放在 half 内：当前 `sA` 同时承担 FC1 A staging，half 0 提前写入的 Q1 结果会被 half 1 的 A TMA
覆盖。先把 BF16 activation 分 half 落到独立 `sC`，可及时结束 accumulator lifetime，又不引入新的 sA buffer。

任何 correctness、work identity 或 synchronization failure 都使该 candidate 无效；此时只能报告实现问题，
不能得出“native N64 是否解决 spill”的结论。

## 4. 环境、case 与证据身份

| Field | Locked contract |
|---|---|
| GPU | dedicated SM120 5KP；同一 GPU UUID；无 foreign process |
| Container | exp_005 canonical container digest |
| Toolchain | CuteDSL 4.6.0；CUDA/nvcc/ptxas 13.2.78；fresh 记录实际版本、module path 与 dependency-tree content hash |
| Compiler mode | production-equivalent；关闭 IKET/marker/compiler-opt overlay |
| Shape | `E=256, H=2048, I_tp=512, topk=8, SwiGLU, BF16 output` |
| Precision | BF16 input；FC1/FC2/Q1 NVFP4 block-scaled |
| Launch | outer CUDA Graph；`num_sms=110`、`max_active_clusters=110`、grid.z=110 |

Cases：

- `M=256`：compile/correctness smoke 与 tail coverage；
- `M=8192`：primary resource、static/dynamic spill case；
- directed `sparse_empty`、`exact_128`、`tail_129`、`hot_expert` route fixtures：覆盖少于 110 个 task、空转 CTA、
  tile/tail 与 hot expert 的退出同步；
- branch/half/slice canary fixture：Gate/Up、N64 half 0/1 和四个 logical N128 slice 使用可区分的 weight/SFB
  payload，独立 reference 验证地址和覆盖，不能只依赖随机输入。

每个 arm 必须使用独立 Python process、独立 overlay import 和 fresh JIT root。manifest 保存 host/GPU UUID、
container digest、driver、`nvcc --version`、`ptxas --version`、CUDA runtime、Python、Torch、CuteDSL version/path、
FlashInfer commit、source/diff/JIT cubin/PTX/SASS hashes、grid/block、fixture/input/output hashes。若两臂 toolchain、
source 或 cubin 身份漂移，停止拼接比较。

## 5. 验证门与采集

### Gate A：实现与结构

- candidate 必须编译、block 为 `(288,1,1)`，W8 仍为 TMA warp；
- 保存 source diff、generated IR/PTX/SASS、descriptor/layout 审计与 cubin identity；
- 证明仅有一对 `partition_shape_C(M128,N64)` accumulator（目标约 `32+32 FP32/thread`），每个 half 后立即
  SwiGLU/accumulator death；用 source/IR def-use 与锁定 cubin 的 `nvdisasm -plr` 证明 half 0 OMMA destination
  不跨 half 1；若映射证据无法闭合，不得宣称“live range 已被证明缩短”；
- descriptor ledger、exact OMMA opcode/count/phase ledger、一次 full-N128 Q1/FC2/scatter 均通过；
- barrier hang、illegal access、full-N128 TMA replay、额外 FC2/scatter 或 work mismatch 均 fail closed。

### Gate B：正确性

- M256/M8192、route fixtures 与 branch/half/slice canary 均使用 formal independent reference；
- 报告 finite/nonzero、shape/dtype、max abs、relative L2、cosine、token relative-L2 p99；
- 至少两次 replay；阈值沿用 exp_005 的预注册 protocol，不在看到数据后放宽。

现有 scheduler exact-once 证据仅为 descriptor multiset、terminal head 与 scheduler source/AST-identical 下的推断，
不是 consumed-bitmap 直接证明；报告必须保留这个证据边界，不能把 output sentinel 当作 exact-once 证据。

### Gate C：静态 zero-spill

先在相同 dependency tree、相同 entry 下 fresh 编译 anchor：必须复现 `224 B/thread` static frame、nonzero
compiler SpillRefill、exact 448 OMMA；否则实验前提未复现，不能宣称 candidate“消除了当前 residual spill”。

随后对每个 distinct candidate cubin 同时要求：

- `STACK=0 B/thread`；
- compiler-annotated SpillRefill local SASS 为 0；
- 记录 registers/thread、SMEM/CTA、block 与理论 occupancy；
- 使用 `nvdisasm -plr` 保存 register lifetime/resource 证据。

若静态 spill 非零，实验直接判定“该 native N64 实现未解决 spill”，无需采 performance 或用 latency 替代。

### Gate D：动态 zero-spill

仅当 Gate A–C 全部通过，才对 anchor/candidate 的 M8192 final CUDA Graph replay node 采 matched NCU：

- 以下四个 metric 必须存在、非 `n/a`；candidate 必须全为 0，anchor 必须复现非零：
  `sass__inst_executed_register_spilling_op_read/write` 与
  `sass__inst_executed_register_spilling_mem_local_op_read/write`；
- SourceCounters/InstructionStats 对应 local PCs；
- executed Tensor instructions、FP4 Tensor ops、registers、stack、achieved occupancy 与 launch geometry。

正式判定要求 compiler spill/refill 与动态 spill traffic 均为 0。若 NCU capture 使用的 cubin/hash 与 Gate C
不同，或缺失 mangled/demangled entry symbol、entry resource record、loaded cubin hash、graph node/launch ID、
grid/block、NCU version/section identity，capture 无效，必须 fresh 重采。`launch__stack_size` 只是 runtime
configured limit，不能替代 cubin static `STACK=0`。

## 6. 决策树与产物

```text
compile/layout/sync fail
  → 实现未闭合；无 spill 结论

correctness/work identity fail
  → candidate 无效；无 spill 结论

fresh anchor 未复现 residual spill / work identity
  → 前提未复现；无“消除 spill”结论

correct + static spill > 0
  → 仅 reject candidate_vN；定位该实现的剩余 live set/PC

correct + static spill = 0
  → matched NCU
      ├─ dynamic spill = 0：接受“该完整实现 bundle 可 zero-spill”的可行性结论
      └─ dynamic spill > 0：继续按 PC/source 定位，不能宣称 zero-spill
```

产物统一放在 `results/`：

- `overlays/`：anchor/candidate immutable source、diff 与 hashes；
- `raw/`：correctness、route/task、compiler/JIT artifacts；
- `ncu/`：仅保存通过静态门后产生的 matched capture；
- `manifest.json`：机器、软件、source/cubin、fixture 与证据身份；
- `result.md`：只报告 correctness、work identity、static/dynamic spill 结论及证据边界。

正式收口前执行一次 ex-post data audit。本实验不测 latency，zero-spill candidate 的性能与 TC cadence 另开后续
实验，避免把可行性验证和性能优化混成一个 bundle。

## Plan Review

- Date: 2026-07-19
- Reviewer: isolated `experiment-plan-review` subagent（含独立 gate/identity audit）
- Verdict: **Gaps → 本节落盘时已一次性修正；不再复审**
- Scope: exp_007 唯一一次 ex-ante review；reviewer 未编辑文件、未执行实验。

审计发现并已修正：

1. 原序列把 Q1 放入 half 内，存在 `sA` 被下一 half A-TMA 覆盖且未锁定 accumulator death 的风险；现锁为
   `N64 Gate+Up → immediate SwiGLU → sC`，两 half 后才做一次 full-N128 Q1。
2. 原计划把 B/SFB 都假定为原生 N64，但 SM120 SFB helper 会物理补到 N128；现分离 FC1-N64/FC2-N128
   contract，B 使用 N64 descriptor，SFB/A/SFA replay 明确为可计数的 allowed consequence。
3. 原 work identity 只有“OMMA 数相容”，无法排除错地址或动态少算；现增加 exact opcode/count/phase ledger、
   descriptor ledger、matched executed Tensor work 与 FC2/scatter/scheduler AST identity。
4. 原计划未要求 fresh anchor 复现 residual spill；现把 anchor 的 224 B frame、nonzero spill 与 448 OMMA
   设为结论前提，并锁定 dependency-tree content hash。
5. 原 route fixtures 不能定向覆盖 Gate/Up、half 与 slice 地址，sentinel 还会被 Phase 0 清零；现增加
   branch/half/slice canary、恢复 sparse-empty，并明确 scheduler exact-once 只是受约束推断。
6. 原 dynamic-zero gate 未锁定 entry 与 exact metrics；现要求四个 spill metric 全部存在、symbol/cubin/node/launch
   身份闭合，missing/`n/a` 一律失败，且不把 `launch__stack_size` 当 static frame。
7. 原决策树会从单一 v0 外推一般性结论；现所有 verdict 都限定到 immutable candidate implementation bundle，
   final 版本必须 fresh 跑完所有 gate，禁止跨版本拼证据。

## Execution Note：canary_v0 幅值失配与预注册 follow-up

第一次 branch/half/slice canary（`canary_up` / `canary_gate`）在两臂的 8 个 N64 block 独立
reference gate 全部通过，block relative-L2 为约 0.18%–0.32%；但该 fixture 的输出幅值远高于 canonical，
导致 candidate↔anchor 和 replay self-drift 的单点 `max_abs=0.25`，超过 exp_005 correctness protocol 的
固定 `0.1` cap。因此 canary_v0 必须保留为 **strict gate failure**，不能因 relative metric 通过而忽略，也不能
事后放宽阈值。

在运行 follow-up 前锁定 `canary_v1`：保持输入、W1 的 Gate/Up/8×N64 marker、FC2 diagonal mapping、route、
block relative-L2 limit 和完整 strict correctness thresholds 不变；唯一 fixture 改动是把所有 expert 的
`w2_global_scale` 固定为 `0.25`，线性缩小最终输出幅值。v1 使用新的 results/JIT root 与显式 fixture manifest，
不得覆盖 v0。v1 的第一支实际在 independent-reference replay 0 失败，说明 `w2_global_scale` 不是当前
harness 中安全的“只缩放最终输出”控制；该 fixture 被立即 reject，未继续拼证据，阈值仍未改变。

随后在运行前锁定 v2：恢复 v0 的全部 input/weight/scale/Q1/FC2 payload，只把
`token_final_scales` 统一乘 `0.25`，即在最终 atomic scatter 权重边界线性缩小输出。v2 使用新的
results/JIT root；只有 v2 两臂的 independent reference、8-block canary、route/task 和 strict
cross-arm/self-drift 全部通过，canary gate 才能闭合。v0 strict failure 与 v1 fixture failure 都进入
manifest 和最终证据边界。
