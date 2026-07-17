# exp_004：Fused Phase Timing Breakdown Plan

Status: **closed — `measurement perturbation prevented formal timing`**

本实验只测量 `MoEDynamicKernel` 内部 phase 的时间分布，不修改 production kernel，不做性能优化。
所有测量代码必须位于 experiment-owned overlay；任何插桩若改变 production 的 resource、spill 或
semantic work，实验立即降级或停止。

## Execution Closure — 2026-07-17

- 目标 5KP identity 已闭合：`GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`，SM12.0，110 SM。
- `normal_no_marker` 与 `measurement_no_marker` 均保持 `REG=255 / STACK=488 B/thread` 和相同 local SASS / semantic projection。
- `probe_candidate` 变为 `REG=255 / STACK=456 B/thread`，local SASS 也漂移；probe replay reference correctness 失败，timing/CTA 写回为 `0/776016` 与 `0/2536`。
- PTX 中能确认 clock/store lowering，但 0-write 的具体原因未定位；IKET audited provider 不可用。
- 按预注册 immediate-stop 收口，不发布 phase share，不继续 phase capture、calibration 或 NCU。

## 1. Goal

在固定的 `M=8192` workload 上，测量完整 task population 中四个 MMA consumer warp 的真实执行时间分布：

1. FC1 Gate；
2. FC1 Up；
3. SwiGLU + FC2-input Q1；
4. FC2 GEMM；
5. FC2 epilogue + route-weighted atomic scatter。

主结果是 **MMA-consumer warp elapsed-cycle share**。Warp 4 的 Gate/Up/Down TMA producer、wait 与
consumer phase 的 overlap 单独报告。P0–P4 的 Clear、Histogram、Prefix、Route/Q0/Pack 和 Task Publish
只作 secondary breakdown，不得增加主插桩密度或阻断 T1–T4 的收口。

本实验必须回答：

- Gate、Up、SwiGLU+Q1、FC2 GEMM、FC2 epilogue/scatter 分别占 complete MMA-consumer
  task elapsed cycles 的多少；
- FC2 的时间主要在 GEMM consumer interval，还是 epilogue/scatter interval；
- Warp 4 的 Down prefetch 与 T3/T4 实际 overlap 到什么程度。

本实验不回答：

- 这些 share 不是各 phase 对 production kernel wall 的可加贡献；
- phase share 不能把 whole-launch NCU counters 按比例拆分；
- phase 占比高不等于该 phase 是 bottleneck；
- “GEMM 前后操作是 memory-bound”是后续待验证假设，不是本实验结论；
- 本实验不证明 register spill 对 latency 或 TensorCore cadence 的因果影响；
- 不比较 CUTLASS Chain，不产生 backend speedup。

## 2. Fixed Case and Production Anchor

| Field | Locked value |
|---|---|
| Arm | `cutedsl_bf16_fused` only |
| Shape | `M=8192, E=256, H=2048, I_tp=512, topk=8` |
| Activation / output | `SwiGLU / BF16` |
| Weights / internal activation | FC1 Gate/Up、FC2 Down 为 NVFP4 block-scaled |
| Fixture | exp_002 `fixture.py`；base seed `2026`，M8192 routed seed `10218` |
| Operator boundary | BF16 input + precomputed top-k ids/weights → one fused W4A4 MoE launch → BF16 output |
| Router boundary | router logits、softmax、top-k selection 不在本 kernel 内 |
| Launch mode | outer CUDA Graph；只选择一个明确绑定的 final replay `MoEDynamicKernel` node |
| Kernel / launch | `MoEDynamicKernel`；grid `(1,1,110)`；block `(160,1,1)`；1 CTA/SM |
| Hardware identity | 仅接受经过 5KP gate 的 `NVIDIA Graphics Device`，SM `12.0/12.1` 且 `110 SM`；记录 UUID/PCI ID 并要求独占 lease、capture 前后无 foreign compute process。`RTX PRO 6000*` 即使为 SM12x 也不得替代 |
| Warp roles | Warp 0–3：MMA consumers；Warp 4：TMA producer |
| Tile | `BM×BN×BK = 128×128×128` |
| FC1 work / task | Gate 16 个 H/K128 tiles；Up 16 个 H/K128 tiles |
| FC2 work / task | one I128 slice；16 个 H/N128 output tiles |
| Publish policy | `full_tile_publish_enabled=0`；P3/P4 与 P5 之间无 route/compute overlap |
| Expected task population | `task_tail=2536`；`task_head=2646=task_tail+grid_z` |
| Routed rows | `row_counts.sum=M×topk=65,536` |

Production source/binary identity 继承 exp_002/exp_003 的 canonical baseline：

| Identity | Locked value |
|---|---|
| Production source SHA-256 | `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106` |
| Production dispatch SHA-256 | `cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4`（`moe_dispatch.py` consumer adapter anchor） |
| Production wrapper SHA-256 | `bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4`（`b12x_moe.py`，只读、不做 overlay） |
| Baseline cubin SHA-256 | `9313fcbc0dd686f0684705e869fdd227608ac83ca43c1dc99d203f8e7143ca79` |
| Baseline SASS SHA-256 | `34b4c38161642a27ca6b4ec41ffad0bd70f6ff99fd8118997a4b2416c5e3abba` |
| Baseline NCU SHA-256 | `3367df71ef0c3b750c03c60436d100a994863271016995c76f21195dd9eaaea8` |
| Registers | `255/thread`（allocated tier 256） |
| Shared memory | `92,160 B/CTA` total；`91,136 B` dynamic；`1,024 B` static |
| Actual compiler stack | `488 B/thread` |
| Spill structure | `122 words/lane = Main 108 + Tail 14` |

Fresh production build 必须重新验证这些 identity。若 pinned toolchain 下无法重现 canonical cubin/SASS，
先解释 toolchain/source drift 并重建 production anchor；禁止将新 binary 与历史 NCU/exp_003 spill 证据拼接。
manifest 必须记录 driver、CUDA runtime、`nvcc --version`、`ptxas --version`、Python、Torch、CUTLASS DSL
版本与 module path、container digest、GPU UUID/PCI ID、clocks/power state、source/overlay/JIT artifact hashes。

## 3. Source Phase Contract

以下 line boundary 只对 production source SHA `94b4dd…` 有效。源码漂移时必须重新审阅，不能机械沿用。

### 3.1 Launch-level phases

| Phase | Source | Ownership | Scope |
|---|---|---|---|
| P0 Clear | `963–1022` | all 160 threads | clear routing/queue/output + grid barrier |
| P1 Histogram | `1024–1036` | all threads | routed-pair expert histogram + grid barrier |
| P2 Prefix | `1038–1055` | `flat_tid==0` computes；others wait | serial expert tile prefix + grid barrier |
| P3 Route + Q0 + Pack | `1057–1430` | all 5 warps, warp-private | route metadata、BF16→FP4 Q0、expert-major GMEM pack |
| P4 Publish | `1432–1486` | CTA leader；`flat_tid==0` writes tail | deferred task descriptors + grid barrier |
| T0 Claim/cache | `1755–1840` | CTA leader claims；Warp 0–3 cache 32 rows each | task handoff and scatter metadata cache |

P0–P4 由 resident-grid barriers 串行分隔，但 secondary probe 仍只报告 instrumented CTA/grid interval；
不得把它除以未插桩 `1759.483 us` 得到 production wall share。

### 3.2 Primary MMA-consumer boundaries

| Primary interval | Start / end source boundary | W0–W3 work included |
|---|---|---|
| `task_envelope` | after scatter-metadata cache sync、consumer branch entry `1844` → after `pass_final.arrive` `2548` | descriptor/setup + complete selected task/slice consumer lifetime |
| `fc1_gate` | `gate_acc.fill` / first Gate wait `1904–1907` → `pass_gate.arrive` `2034` | producer wait visible to consumer、S2R、Gate OMMA |
| `fc1_up` | `up_acc.fill` / first Up wait `2038–2041` → last Up OMMA `2167` | producer wait visible to consumer、S2R、Up OMMA |
| `swiglu_q1` | activation entry `2168` → post-Q1 MMA-only barrier `2302` | alpha、SwiGLU、BF16 sC、Q1、FP4/SFA sA |
| `fc2_setup` | FC2 activation/scale hoist `2325` → hoist complete `2350` | sA/sSFA S2R once per task；单列，不并入 GEMM |
| `fc2_gemm[j]` | output tile wait `2352` → last Down OMMA `2408` | pipeline wait、B/SFB S2R、Down OMMA |
| `fc2_epilogue_scatter[j]` | `2410` → post-scatter barrier complete `2544` | down scale/convert、sC store、barriers、route-weight multiply、atomic scatter |

每个 complete `(task_slot, slice, warp_id)` 必须恰有一个 Gate、一个 Up、一个 SwiGLU+Q1、一个 FC2 setup，
以及 `16` 个 FC2 GEMM 和 `16` 个 FC2 epilogue/scatter interval。源码运行时 setup、phase transition 与
未覆盖空隙统一进入 `residual/task_control`，不得静默分摊给相邻 phase。

### 3.3 Warp 4 producer boundaries

| W4 interval | Source | Meaning |
|---|---|---|
| `gate_tma` | `2584–2612` | Gate A/SFA + W/SFB TMA issue |
| `gate_pass_wait` | `2614–2616` | wait for all MMA warps to complete Gate |
| `up_tma` | `2620–2656` | Up A/SFA + W/SFB TMA issue |
| `down_tma` | `2658–2699` | continuous FC2 Down W/SFB prefetch |
| `final_pass_wait` | `2701–2703` | wait until FC2/scatter releases task buffers |

Warp 4 的 `down_tma` 可以与 W0–W3 的 Up tail、SwiGLU+Q1 和 FC2 consumer work overlap。
W4 durations 只用于 producer timeline、wait 与 overlap，不进入主 phase-share 分母，也不能与 W0–W3
duration 相加。

## 4. Timing Denominator and Coverage

### 4.1 Primary denominator

对每个 complete MMA consumer warp-task：

```text
task_elapsed(t,w) = task_end_tick(t,w) - task_start_tick(t,w)

phase_elapsed(p,t,w) = union of non-overlapping intervals for phase p

share(p) = Σ(t,w) phase_elapsed(p,t,w)
           / Σ(t,w) task_elapsed(t,w)

residual(t,w) = task_elapsed(t,w)
                - union(all declared primary intervals within that same warp-task)
```

其中 `w∈{0,1,2,3}`。四个 warp 同时工作，因此该分母是 **MMA-warp elapsed cycles**，不是 CTA wall，
也不是 kernel wall。按 warp 相加是在统计 consumer-role time；它有意保留四个并行 warp 的 multiplicity。

只有同一 warp 内的 mutually exclusive intervals 可以构成 100% split。以下量必须分开：

- `share(p)`：可加的 MMA-consumer task-cycle share；
- W4 producer/consumer overlap：同一 CTA/task 的 interval intersection / union，非可加；
- full-grid phase coverage：若 fallback provider提供 common timestamp，只是 wall coverage，可能互相 overlap；
- production kernel wall：只来自未插桩 benchmark/NSys，不按 phase 拆分。

### 4.2 Complete population gate

Primary clock probe 目标是完整 population，不做抽样外推。固定 workload 期望：

```text
2536 task slots × 4 MMA warps = 10,144 complete consumer warp-tasks
```

Formal result 要求：

- 每个 task slot 恰好出现一次，descriptor 与 pinned task table 一致；
- 每个 task slot 的 W0–W3 都有完整、单调、无重复的 boundary；
- 每个 warp-task 有 16 个 FC2 tile pair；
- intervals 均包含于 task envelope；同一 warp 内无非法 overlap；
- `declared union + residual == task envelope`；
- complete consumer warp-task coverage 为 `100%`；missing/duplicate event 不解释为 0；
- 记录实际 `CTA z → task_slot` assignment、每 CTA task count、full/partial `valid_rows` 和 completion skew；
- 至少 5 次独立 measured replay；逐 phase share 的最大 run-to-run absolute drift `<=1%`，否则报告
  distribution 但不输出单一 canonical share。

若只能得到 selected-CTA capture，必须改为预声明的 stratified estimate：`task_slot early/steady/tail ×
full/partial valid_rows × slice_count`。每个非空 stratum 至少覆盖 8 个 complete tasks、来自至少 3 个 CTA
captures，并按完整 task population weighting；coverage 不足只能判 `inconclusive`，不能把 selected CTA
推广到 full grid。

## 5. Primary Measurement Method

### 5.1 Sparse `%clock64` boundary probe

主方法是在 experiment-owned overlay 中加入低密度、lane-0-only 的 `%clock64` boundary records：

- W0–W3 在上述 primary boundaries 读取 clock；W4 在 producer boundaries 读取 clock；
- timestamp 读取后立即写入预分配的 experiment timing buffer，不把 start timestamp 作为长生命周期
  register 跨 phase 保留；
- event index 由 `(replay, task_slot, slice, warp_id, phase, output_tile, edge)` 唯一确定；不使用 runtime
  atomic append；
- 每个 event 保留 raw start/end tick，而不是只保留 kernel 内聚合值；
- `%clock64` 只用于同一 SM/CTA/warp 的差值；禁止跨 SM 直接比较绝对 tick；
- W4 与 W0–W3 同属一个 CTA/SM，可以在同一 task 内计算 interval overlap；
- P0–P4 secondary records 只允许 CTA leader 采集，且必须在 primary candidate 通过后作为独立 gated build，
  不得增加 primary binary 的插桩密度。

#### 5.1.1 Locked probe ABI and slot layout

Probe 不修改 production tree；experiment 同时生成 `moe_dynamic_kernel.py` 与 `moe_dispatch.py` 的完整模块
overlay。后者只负责 timing workspace allocation、runtime pointer/shape plumbing 与 compile-time
`probe_enabled` 选择；`b12x_moe.py` 保持 byte-identical production source。三身份的 kernel/dispatch
overlay hash 和两份 diff 都进入 binary identity，不能只登记 kernel source。

每次 replay 使用一个预分配 `int64` timing tensor，device 侧按 `uint64` 写 `%clock64`，host 侧以 `-1`
作为未写 sentinel。另有 `int32 task_cta_z[task_capacity]`，同样以 `-1` 初始化，只允许 W0 lane 0
为被 claim 的 task 写一次实际 `bidz`。固定 interval layout 为：

```text
consumer interval[37]:
  0 task_envelope, 1 fc1_gate, 2 fc1_up, 3 swiglu_q1, 4 fc2_setup,
  5..20 fc2_gemm[0..15], 21..36 fc2_epilogue_scatter[0..15]

consumer_tick(task, warp, interval, edge)
  = ((((task * 4) + warp) * 37 + interval) * 2 + edge)

w4 interval[5]: gate_tma, gate_pass_wait, up_tma, down_tma, final_pass_wait
w4_tick(task, interval, edge)
  = task_capacity * (4 * 37 * 2) + ((task * 5 + interval) * 2 + edge)

timing_ticks_capacity = task_capacity * (4 * 37 * 2 + 5 * 2)
```

`edge=0/1` 分别是 start/end。M8192 actual `task_tail=2536` 时 formal expected writes 为
`2536 × 306 = 776,016 uint64 = 6,208,128 B`；workspace 按 runtime `task_capacity` 分配，tail 之外
必须保持 sentinel。每次 measured replay 在 graph 外 fill sentinel 并同步，graph 只包含目标 launch；replay
同步后立即保存 raw tensor。`measurement_no_marker` 必须保持整个 buffer 和 CTA map 为 sentinel；
`probe_candidate` 必须 exact-fill tail 内全部 expected slots，且无 duplicate/negative/non-monotonic interval。
不从 raw duration 中减 calibration；calibration 只用于 perturbation 上界。

FC2 split 需要每个 output tile 的 `gemm_begin/gemm_end/scatter_end`，但不得恢复旧 IKET 实验的
per-OMMA marker。计时 buffer 的额外 store bytes、event count 与空 marker calibration 必须显式记录。

### 5.2 Replay protocol

1. 用 pinned fixture 创建 input/reference；完成一次 warmup/graph capture。
2. 在每次 measured replay 前重置 experiment timing buffer；reset 不得成为目标 kernel 内工作。
3. 每次 run 只选择一个明确的 CUDA Graph replay node，并绑定 `(pid, context, graph, grid, kernel)`。
4. 保存 raw timing buffer、task workspace、correctness、external CUDA-event latency 与 binary identity。
5. 至少 5 次 measured replay；capture 顺序固定，记录 clocks/power/process state。
6. timing candidate 只用于 phase share；production latency 继续使用 normal/no-marker arm。

## 6. Three Binary Identities

每个正式 capture 必须有三套 fresh JIT identity，禁止复用 cache：

| Identity | Purpose | Required relationship |
|---|---|---|
| `normal_no_marker` | exact production timing/resource anchor | 必须重现 canonical production source/binary/resource |
| `measurement_no_marker` | 与 probe 相同 overlay、kernel ABI、compiler path，probe compile-time disabled | 除未使用 measurement plumbing 外，semantic binary 必须与 production 等价 |
| `probe_candidate` | 同 measurement build，clock records enabled | 只允许声明的 clock/predicate/address/probe-store 指令差异 |

比较关系：

1. `normal_no_marker ↔ measurement_no_marker`：必须通过 production equivalence gate；
2. `measurement_no_marker ↔ probe_candidate`：必须通过 instrumentation perturbation gate；
3. `normal_no_marker ↔ probe_candidate`：只用于判断 candidate 是否仍代表同一 production resource/work tier，
   probe duration 不作为 production latency。

任一 controlled invariant 失败都不得在运行后放宽；需要新 plan revision 和 fresh build/capture。

## 7. Fail-Closed Perturbation Gates

### 7.1 Source and overlay

- production source SHA 必须为 `94b4dd…`；
- overlay diff 只包含 timing buffer plumbing、declared timer read/store、event indexing 和注释；
- 反向移除 probe patch 后必须 byte-for-byte 恢复 production source；
- 不得改变 compute、memory、barrier、pipeline、tile、warp layout、task queue、publish policy 或 launch geometry；
- production tree 始终保持未修改，overlay 使用独立 import/JIT root。
- kernel 与 consumer-adapter dispatch 分别使用 exact-module overlay；normal arm 的两份 overlay 都必须
  byte-identical production，measurement/probe dispatch 只能增加 timing workspace/ABI plumbing，wrapper
  `b12x_moe.py` 不得改变；任何未登记的 dispatch/backend-selection/cache-key 变化立即停止。

### 7.2 Resource and spill identity

三套 binary 必须保持同一 resource tier，candidate 尤其必须满足：

- `REG=255/thread`，allocated tier 不变；
- `STACK=488 B/thread`；
- static/dynamic/total SMEM 与 occupancy tier 不变；
- Main bundle 仍为 `54×STL.64 = 108 words/lane`，stack slot set、producer→store→reload→first-use
  顺序不变；
- Tail bundle 仍为 14 words/lane，5 个 second-pass accumulator register values 与 9 个 scalar 的
  save/reuse/restore 链不变；
- compiler spill/refill annotation、dynamic local load/store word/sector count 与 baseline 闭合；
- probe 不得新增其他 STL/LDL、local memory、register demotion 或不同 occupancy tier。

旧 cadence 实验已经证明“REG 相同”不足：IKET marker 令 stack 从 `488→432 B/thread`，因此该实验的
phase 数值全部失效。本实验只要出现任意 stack/spill drift 就 fail closed，即使 candidate latency 更快。

### 7.3 Semantic SASS and CFG

剥离明确登记的 probe-only instructions/blocks 后，fresh baseline、measurement control 与 candidate 必须保持：

- non-probe CFG topology 和 branch targets；
- semantic opcode sequence/order；
- exact OMMA、UTMALDG/TMA、LDSM、BAR、atomic/reduction、non-probe LDG/STG counts；
- Gate/Up/FC2 每 phase 的 expected Tensor instruction count 与 ordinal；
- pipeline wait/release、`pass_gate`、`epilog_sync`、`pass_final` 位置；
- task claim、descriptor、scatter atomic semantics。

历史 baseline 的 selected static projection 为 `OMMA=896, UTMALDG=40, LDSM=200, BAR=34,
ATOMG=9, REDG=4, LDG=53, STG=75`；fresh extraction 必须与 canonical artifact核对。Probe stores 不计入
semantic STG，但必须单独计数。绝对 PC 可以因 probe 插入而移动；比较使用 normalized instruction order、
stack slot 和 producer/consumer relation，不能用 PC shift 掩盖 semantic drift。

### 7.4 Logical and dynamic work

- kernel/grid/block/cluster 与 graph node identity 固定；
- `task_tail=2536`、`task_head=2646`、`row_counts.sum=65,536`；
- task descriptor table、valid rows、expert/m-tile/slice population 与 control 一致；
- 每个 task 完成 Gate、Up、SwiGLU+Q1、16 个 FC2 output tiles 和 scatter；
- dynamic Tensor instructions 与 FP4 Tensor ops 和 production anchor一致；
- timing event count 必须与 task/warp/output-tile 公式闭合；
- actual persistent CTA assignment 允许因原子 claim 有 run-to-run variation，但必须完整记录；若 per-CTA task-count
  distribution 或 tail skew 超出 no-marker repeat envelope，则 candidate 只能 diagnostic。

### 7.5 Correctness

`normal_no_marker`、`measurement_no_marker` 与 `probe_candidate` 各自通过同一 quant-aware oracle：

- output shape/dtype 与 reference 一致；
- finite、每 token nonzero；
- cosine `>=0.999`；
- relative-L2 `<=0.02`；
- max-abs `<=0.08`；
- candidate 相对 measurement control 的逐 token relative-L2 p99 不超过 control 上界 `+0.005`；
- `token_map/token_weights`、row counts、task workspace 和 routing weights 满足 pinned invariants；
- atomic output 不要求 bitwise equality，但禁止仅用总 checksum 代替 element gate。

### 7.6 Runtime probe overhead

- 保存同一工具链 candidate/control 的 external CUDA-event latency repeats；
- candidate 相对 measurement control 的 median latency drift必须 `<=1%`，且不得超过 control repeat spread
  的 5 倍；
- back-to-back empty clock/store boundary calibration 的 total p95 upper bound 必须 `<=1%` of covered task ticks；
  consumer declared durations 按实际 start-side probe stores 计数：每个 warp-task 为
  `Gate 1 + Up 2 + SwiGLU 2 + setup 1 + 16×(GEMM 1 + epilogue 2) = 54`，不能只按 36 个逻辑 interval 低估；
- timing buffer 不得 overflow，event slot 不得 overwrite/duplicate；
- 若 resource/SASS/work/correctness 全过但 runtime overhead gate失败，结果只能标
  `instrumented diagnostic share`，不能写成 production-representative share。

## 8. Fallback Measurement Path

如果 sparse clock probe 因 DSL codegen、buffer plumbing 或 capture feasibility 无法工作，允许一次独立 fallback：

### Coarse role-local IKET

- 只保留 `task_envelope`、Gate、Up、SwiGLU+Q1、FC2 per-tile GEMM、FC2 per-tile epilogue/scatter 和 W4
  producer ranges；禁止 per-OMMA、per-S2R、per-wait marker；
- 使用 audited IKET provider/version 与 KDK `iket_safe_capture.py`，fresh output、bounded context buffer、
  explicit repeated-workload acknowledgement；
- 首选 full-grid coarse capture；先估算 event volume 并满足 provider/device memory cap；
- full-grid 不可行时只能使用预声明 selected CTA 顺序和第 4.2 节的 stratified coverage gate；
- IKET 自己建立 `normal/no-marker → IKET-compiler/no-marker → coarse-marker candidate` 三身份；
- 完整复用第 7 节 resource/spill/SASS/work/correctness gates；旧 `488→432` 问题不得豁免；
- timestamp unit 未由 decoded artifact证明时只写 `raw timestamp units`；share 可使用同一线性时钟的无量纲比例，
  但不得把绝对值写成 cycles/ns；
- IKET instrumented duration 不作为 production latency，也不按 phase share拆分 NCU metrics。

如果 clock probe 与 coarse IKET 都无法通过 perturbation gate，则实验正式结论是
`phase timing blocked by measurement perturbation`。可以对 unmodified production cubin 做 NCU/PC sampling，
但它只能提供 phase-mapped sample/stall distribution，不能替代 elapsed-time share，也不能强行收口本 Goal。

## 9. Raw and Derived Artifact Schema

所有实验产物位于本目录 `results/`；raw provider artifacts 保持只读，derived artifacts 可重建。

### 9.1 Minimum raw event table

`results/raw/phase_events.csv` 每行一个 closed interval：

| Column | Meaning |
|---|---|
| `run_id` | unique measured replay |
| `source_sha256 / cubin_sha256 / sass_sha256` | exact candidate identity |
| `gpu_uuid / clock_state` | device/run identity |
| `kernel / graph_launch_key / grid_id` | exact target launch |
| `cta_z / task_slot / expert / m_tile / slice / valid_rows` | persistent task identity |
| `warp_id / role` | `0..3/mma_consumer` or `4/tma_producer` |
| `phase / subphase / output_tile` | declared semantic interval |
| `start_tick / end_tick / timestamp_unit` | raw timestamps；不预先减 calibration |
| `complete / duplicate / overflow` | event validity |
| `provider_scope` | full-population clock、full-grid IKET or selected CTA |

另保留：

- `results/raw/task_population.csv`：2536 个 expected/observed descriptors、valid rows、actual CTA assignment；
- `results/raw/binary_identity.json`：三 binary、toolchain、resource、SASS/CFG 和 spill gates；
- `results/raw/correctness.json`、`work_identity.json`、`latency_repeats.csv`、`calibration.csv`；
- provider-native clock buffer 或 IKET decoded JSON 与 SHA-256 manifest。

### 9.2 Derived tables

`results/derived/mma_phase_share.csv`：

```text
phase, role, runs, complete_warp_tasks, intervals,
sum_ticks, denominator_ticks, share_pct,
per_warp_task_p50, per_warp_task_p95, run_spread_pct,
coverage_pct, residual_pct, verdict
```

`results/derived/w4_overlap.csv`：

```text
producer_phase, consumer_phase, task_count,
producer_union_ticks, consumer_union_ticks, intersection_ticks,
producer_covered_pct, consumer_covered_pct, p50, p95, scope
```

`results/derived/secondary_p0_p4.csv` 只在 secondary build 独立通过 gates 后生成，报告 CTA-level
elapsed distribution / grid interval coverage；不得与主 task share 合并。

最终 `results/result.md` 只展示：证据身份和 gate、主 MMA phase share、W4 overlap、secondary P0–P4（若合法）、
限制与下一步。Memory/compute-bound 判断必须留给后续 production NCU/PC/SASS 分析。

## 10. Acceptance and Stop Conditions

### Formal acceptance

只有同时满足以下条件，才能发布 canonical phase share：

1. production source/binary identity闭合；
2. 三 binary 的 source/resource/spill/semantic SASS gates 全部通过；
3. work/task/Tensor/event-count closure 全部通过；
4. correctness 全部通过；
5. timing buffer/provider无 overflow、missing、duplicate 或 timestamp-order failure；
6. full population coverage 100%，或 selected-CTA stratified estimate 满足预声明 coverage；
7. runtime overhead 与 repeat stability gates 通过；
8. every reported number 可追溯到 exact run/cubin/task/warp/unit。

### Immediate stop

出现以下任一项立即停止 formal interpretation：

- production source/hash、kernel/grid/block/graph、toolchain或GPU identity漂移；
- `REG!=255`、`STACK!=488`、SMEM/occupancy tier变化；
- Main 108 或 Tail 14 spill chain变化，或新增 local spill；
- normalized non-probe SASS/CFG、Tensor/TMA/barrier/atomic work变化；
- correctness/task/workspace/Tensor work失败；
- probe/IKET trace overflow、event缺失/重复、timestamp unit/scope无法界定；
- selected CTA coverage不足；
- candidate persistent scheduling/tail明显超出 no-marker repeat envelope；
- 试图将 instrumented share 写成 production kernel wall contribution，或从 phase share推导 memory-bound。

停止后保留 raw artifacts 与失败 gate，结论写 `inconclusive` 或
`measurement perturbation prevented formal timing`，不得补造 phase 百分比。

## Plan Review

**Date**: 2026-07-17
**Reviewer**: llm fallback（subagent thread limit；single-round）

**Verdict**: ⚠️ Gaps（已一次性修入本 revision，可直接进入实施，不重复 review）

**Gaps + suggested fix**:

- timing buffer 无法只靠 kernel overlay 接入：将 `moe_dispatch.py` 明确纳入 experiment-owned
  consumer-adapter overlay，并锁定 kernel/dispatch/wrapper 三份 production identity 与 diff 边界。
- 原计划没有可执行的 event capacity/slot/sentinel/reset closure：锁定 37 个 consumer interval、5 个 W4
  interval、runtime-capacity 公式、M8192 expected write count、CTA assignment 与 graph 外 reset gate。
- `SM120/121` 不足以约束目标硬件：新增 5KP identity/110-SM/UUID/PCI/独占 lease gate，并明确禁止
  RTX PRO 6000 代替。
