# exp_011：Fused Route/Q0/Pack 与 Chain Prefix→Expand 瓶颈分析

## 状态

已收口（2026-07-19）：identity/no-marker fidelity、8 组 capture correctness/workspace/event
gate、actual Chain anchor、NCU identity 与 source hash 均闭合。接受 equal-scale
shared-input composite benefit 和 `pair_head` 低优先级；row-allocation atomic 独立贡献、
shared 内部收益拆分及 baseline Fused-vs-Chain 原因保持 unresolved。数据审计与轻量因果审计
均已完成，结论见 `results/result.md`。Plan Review 只执行了约定的一轮。

## 目标

在 `M=8192, E=256, H=2048, topk=8, tile_M=128` 主用例上回答：

1. fresh production Fused `P3 Route + Q0 + Pack` 是否仍稳定在约 `348.80 us`；
2. 能否用实验目录内的 full-kernel identity clone 保留 production 的工作量、数据流、资源与
   P3 时间特征，并作为只改 P3 的受控实验基线；
3. 为什么 Fused P3 比 actual CUTLASS `expandInputRowsKernel` 的既有 `245.60 us` 更慢；
4. 差距主要来自哪类已验证机制：
   - `pair_head` persistent batch claim 与每 batch CTA barrier；
   - `expert_write_rows` 原子分配及其地址依赖；
   - routed-pair 粒度重复读取、量化和写回；
   - warp/CTA work mapping、尾部不均衡或 latency hiding；
   - FP4 quant、scale layout 或 GMEM load/store 的指令与访存效率。

本实验只定位优化点，不直接修改 production。`M=256` 最多作为小 M sanity control，不能取代
M8192 主结论。

## 已锁定源码事实与既有锚点

### Production Fused P3

- 源码：
  `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py`；执行前重新记录
  source hash，并确认实际 specialization 使用 `share_input_across_experts=False`。若实际配置为
  `True`，立即停止，重新审查本计划的 work ledger。
- P3 位于 Fused 自己的 Histogram/Prefix 之后；默认 `full_tile_publish_enabled=0`，P3/P4 完成并经过
  resident-grid barrier 后 compute 才开始，当前没有 route/compute overlap。
- launch 为 `110 CTA × 160 threads`，即 W0–W4 共 5 warps；P3 中 5 个 warp 都是 producer。
- 每个 CTA leader 通过 `pair_head` 一次 claim `5 warps × 2 pairs/warp = 10 pairs`，随后 CTA
  barrier；每个 warp 顺序处理 2 个 routed pairs。
- 每个 routed pair 由 lane 0：
  1. 读取 `expert_id/token_idx/weight`；
  2. 对 `expert_write_rows[expert]` 做一次 global atomic add；
  3. 生成 `phys_tile/phys_row`；
  4. 写 `token_map` 与 `token_weights`，再通过 warp shuffle 广播 metadata。
- 每个 warp 随后处理该 routed row 的 `H/16=128` 个 scale blocks；每 lane 处理 4 个 block，
  每 block 读取 16 个 BF16、求 absmax、做 NVFP4 quant，写 8-byte packed FP4 与 1-byte scale。
- P3 尾部包含 `CTA sync → threadfence → CTA sync → resident-grid barrier`；后续 deferred task
  publish 属于 P4，不得计入 P3。
- exp_004 diagnostic anchor：P3 `348.80 us`，占其 SM-equivalent whole-kernel wall `19.10%`；
  该值是 `%globaltimer` additive estimate，不是独立 CUDA-event kernel latency。

### M8192 逻辑工作量

在所有 top-k routes 均属于本地 256 experts 的当前用例中：

| Work item | 理论值 |
|---|---:|
| tokens | 8,192 |
| routed pairs / physical rows | 65,536 |
| P3 productive batch claims | `ceil(65,536 / 10) = 6,554`；退出 claim 另按动态证据统计 |
| expert-row allocation atomics | 65,536 |
| FP4 scale blocks | 8,388,608 |
| BF16 values quantized | 134,217,728 |
| logical BF16 input read payload | 268,435,456 B |
| packed FP4 row stores | 67,108,864 B |
| scale-byte stores | 8,388,608 B |
| token-map stores | 262,144 B |
| token-weight stores | 262,144 B |

上表是 logical ledger，不冒充 NCU 实际 DRAM/L2 traffic；cache line、重放、write allocation 和
metadata traffic 必须由动态 counters 单独报告。

### Actual CUTLASS Chain

- 路由准备实际由三个 prefix kernels 生成 expert-major row maps：
  `blockExpertPrefixSumKernel → globalExpertPrefixSum{Large}Kernel →
  mergeExpertPrefixSumKernel`。
- `expandInputRowsKernel<BF16, NVFP4, NVFP4 block scaling, non-prequant>` 随后读取
  `permuted_row_to_unpermuted_row` 与 `expert_first_token_offset`，对每个 routed row 读取 BF16、
  量化并写 expert-major FP4 activation、scale 及 permuted route scale。
- 既有 M8192 trace 中 Prefix 三 kernel 区间并集 `46.655 us`，Expand `245.600 us`；Expand
  launch 为 `grid=880, block=256`。这些只用于制定 fresh capture 预期，不直接复用作最终结果。
- actual Prefix/Expand 使用 PDL wait/completion 与真实 graph predecessor；fresh capture 必须保存
  correlation、launch flag、grid/block、模板 specialization 和 loaded cubin identity。

## 比较约定与合法结论

| ID | Subject | Reference | 比较类型 | 允许结论 |
|---|---|---|---|---|
| R1 | B full-kernel identity clone | A fresh production P3 | fidelity-only | 判断实验 clone 是否保留 production P3 embedding；通过前不解释 production 瓶颈 |
| R2 | B P3 | C2 actual Expand | component-reference | 比较量化/pack 组件的绝对时间与 work/traffic；不称为 paired speedup |
| R3 | fresh Fused P1+P2+P3 | C1 actual Prefix3→Expand | stage-context | 仅并列展示 route-map+quant/pack 的 raw time；责任仍不完全等价，不报告 ratio |
| R4 | D full-kernel controls | B identity clone | diagnostic-only | 只支持对应机制相关性；只有未插桩 full-kernel 同向改善才进入优化候选 |

不能把以下量混成一个 speedup：

- Fused `P3` 已消费 Fused P0–P2 的 row counts/tile base，但自身还做动态 row allocation、route
  metadata store、Q0 和 Pack；
- Chain `Expand` 已消费 Prefix3 生成的完整 row maps，只做 routed-row expansion/quant/pack；
- Chain `Prefix3+Expand` 包含 upstream map construction，而 Fused `P3` 不含 P1–P2；
- 两边 physical row order、FP4 scale layout、scheduler 和 kernel boundary 不同。

Fused P0 还负责 scatter output/queue clear，P4 还负责 compute-task publish；Chain Prefix3+Expand
没有这两项对应责任。因此 R3 只使用 P1+P2+P3，并只并列 raw time，不计算 ratio。只有完整算子
未插桩 benchmark 才能使用正式 speedup 语言。

## 共同 logical boundary

所有 arm 从同一个 canonical fixture 派生：

```text
BF16 input [M,H]
+ topk expert ids [M,8]
+ topk weights [M,8]
+ input global/per-expert scale configuration
        ↓
logical routed records:
  (expert_id, token_id, route_weight_bits, occurrence,
   128 × {packed FP4 block, scale byte})
```

比较前把各自 physical output canonicalize 为上述 logical records：

- Fused 用 `token_map/token_weights + expert tile/row metadata` 反解；它没有保存 topk slot，因此
  canonical key 使用 `(expert, token, weight_bits)` 的 occurrence-indexed multiset；
- Chain 用 `permuted_row_to_unpermuted_row + expert_first_token_offset + permuted scales` 反解；
- 主性能 fixture 先断言每 token 的 expert id 唯一；duplicate-expert correctness case 仍按 occurrence
  multiset 校验，不要求恢复原 topk slot；
- 不要求 physical row order、scale swizzle 或 raw storage bitwise 一致；
- 若两边 quant fast-math/rounding contract 不同，各自与自己的 FP32/NVFP4 reference 校验，跨 arm
  只比较 dequantized logical values 的预声明容差，不用误差差异解释性能。

R2 的最近共同输入只到 BF16/token/routes。由于 B 还执行 route allocation 而 C2 已有 row map，R2
仍是非等价组件参照。R3 比 R2 更接近 route-map+quant/pack 的高层责任，但 Fused/Chain 仍有不同
metadata、layout与同步责任，因此只并列 raw time，不是同 kernel speedup。

## Arm 与测量边界

### A. Fresh production P3 anchor

复用 exp_004 已审计的 phase ABI，在未修改 production source 的 matched build 上 fresh capture：

```text
P0 Clear/init → P1 Histogram → P2 Prefix → [P3 Route/Q0/Pack] → P4 Publish
                                            ^ start            ^ end
```

必须同时保存：

1. 5 replay 的 P3 raw additive cycles/ns、`/110` SM-equivalent time、run-to-run spread；
2. no-marker 与 probe whole-kernel CUDA-event latency，量化 instrumentation perturbation；
3. 每 CTA P3 entry/exit、productive/terminal claim 数、pairs/CTA、pairs/warp、专家 row histogram；
4. `pair_idx→(expert,token,weight,phys_row)`、row counts、expert tile base 及 canonical output；
5. 同一次 capture 的 P1、P2、P3 additive raw time；其和仅供 R3 stage-context 使用。

若 fresh P3 与既有 `348.80 us` 的差异超出 paired spread + probe perturbation acceptance band，先
定位 source/compiler/cache 漂移，不继续用旧锚点。

### B. Full-kernel identity clone

在实验目录复制 production kernel/dispatch 为实验 overlay，不修改 production 文件：

- identity arm 保留完整 P0→FC2/Scatter 所有 production phase、`grid=110, block=160`、P3
  5-producer-warp mapping、claim/barrier/atomic/shuffle、FP4 quant/pack layout与原始调度；
- 只加入与 A 同 ABI 的 P3 start/end 观测；后续 compute/Scatter 不删除、不 early-exit，避免编译器
  因缩短 kernel 改变 register lifetime、SMEM overlay与 embedding；
- no-marker identity arm 除 source path/实验 dispatch 外与 production 逻辑相同，用于未插桩 e2e
  对照；probe identity arm只用于 P3 phase；
- 每个 control 使用单独源码/单独编译，不加入 runtime mode 分支，避免控制分支改变 registers/SASS；
- output clear、fixture build、canonicalization 和 correctness comparison 均在 timed boundary 外。

B 必须分别报告 P3 additive phase time与完整 fused launch event time，二者不能互换。

### C. Actual CUTLASS anchors

从 actual CUTLASS BF16→NVFP4 MoE graph fresh capture，不以自写近似 kernel 替代：

- `C1_prefix_expand`：Prefix 三个 kernels 第一个 GPU start 到 Expand completion 的实际区间；同时
  报告各 kernel duration、区间并集、PDL overlap 与中间 gap；
- `C2_expand`：actual `expandInputRowsKernel` 单 launch duration；锁定模板参数、grid=880、
  block=256、PDL attribute、predecessor correlation 和 cubin/SASS；
- `C_timing` 使用 public actual graph：锁定 whole-op correctness、Prefix/Expand correlation、PDL、
  exact cubin/kernel identity与时间，但 public runner 不暴露中间 row map/expanded rows；
- 只有实现 exp-owned exact binding、并从同一个 actual cubin/workspace 导出 row maps、expert offsets、
  expanded FP4 rows/scales/route scales后，才建立 `C_intermediate` correctness。若做不到，允许
  source-shape intermediate harness，但必须标为 descriptive，不能冒充 actual C2；
- 两类 C 都锁定 `fast_math`、4-over-6、per-expert/global scale shape与 BF16→NVFP4 specialization。

若 exact actual launcher、模板或 predecessor identity 未闭合，`C_timing` 失败；不得把既有
`245.600 us` 当 fresh actual 结果。缺少 `C_intermediate` 只限制跨实现 logical-record 对照，
不否定 exact public timing anchor。

### D. Full-kernel 最小 controls

只在 B 通过 R1 fidelity 后执行，顺序固定，不做无界 sweep。每个 control 同时生成 probe 与
no-marker 版本：机制判断以 P3 phase + NCU 为主，是否具有优化价值以未插桩 full-kernel event
改善为准；两者方向冲突时不得接受优化。

1. `shared_equal_scale`
   - 主 fixture 的 `w1_global_scale[E]` 必须先逐 bit 证明全 E 相等；将它规范化为 scalar，使现有
     `share_input_across_experts=True` specialization 生效；
   - 复用 production 已有 quantize-once/fanout 路径：量化值数从 134,217,728 降为 16,777,216，
     logical BF16 input payload 从 268,435,456 B 降为 33,554,432 B；packed FP4/scale/route stores
     保持不变；productive claims 变为 `ceil(8192/5)=1639`；
   - 该 arm 同时把 claim 从 routed-pair 改成 token 粒度并改变 route scheduling，只能称为
     **shared-input composite benefit**；没有 token-schedule+requantize8 companion 时，不把收益
     单独归因成纯量化收益；结论只适用于 shared/equal input scale。

2. `static_batch_schedule`
   - 用确定性的 CTA-round/warp 映射替换 `pair_head` atomic claim；
   - 保留每 round 10 pairs、每 warp 2 pairs、相同 CTA barriers、动态 expert-row allocation、
     Q0/Pack、工作量和输出；
   - 若 latency 与 claim-related atomic/warp-stall 同向下降，只支持“global claim/scheduling 相关”；
     地址顺序变化必须通过 histogram 与 traffic 审查。

3. `precomputed_phys_row`
   - 从 A 的 canonical route fixture 在 timed boundary 外构造 `pair_idx→phys_row`；
   - 保留 B 的 pair claim、token/weight stores、Q0/Pack 和目标物理布局，仅以一次 indexed load
     替换 `expert_write_rows` atomic 及其 dependent address chain；
   - 该 arm 引入一笔 mapping load，不能把时间差直接称为 atomic latency；只有配合 dynamic atomic
     sectors、scoreboard/serialization 与地址分布闭合后才支持 row-allocation 机制。

以上 controls 始终保持 full-kernel `110×160` launch。若仍需 `880×256 direct_row_q0_pack`，它只能
作为独立 `E_source_shape` descriptive arm，不参与 R1/R4、full-kernel e2e fidelity 或 production
优化收益判定。

若当前 controls 不能区分机制，下一 arm 必须由已观察的 PC/data-edge 证据驱动；不得删除必要工作、
跳过同步或降低正确性来制造“优化”。

## Fidelity Gate

### A/B production fidelity

B 只有同时满足以下条件才可代表 production P3：

1. canonical input、route multiset、expert histogram、65,536 routed rows、8,388,608 quant blocks、
   output records 与 A 闭合；
2. exact launch geometry、5-warp producer role、claim batch、barrier/threadfence、P3 timing ABI 相同；
3. source/IR/SASS 中关键 mechanism闭合：BF16 vector loads、absmax/FP4 convert/pack、scale/store、
   pair-head atomic、expert-row atomic、shuffle 与 barrier；无意外 OMMA、TMA、spill/refill 或 runtime
   arm dispatch；
4. exact kernel entry 的 registers/thread、static/dynamic SMEM、stack、occupancy limit、cluster
   limit 与 production 一致；identity clone 的关键 SASS/resource/cubin section 必须闭合，不能靠
   padding 伪装；
5. A/B 都使用同一 `%globaltimer` additive denominator 和固定 5 replay。阈值在运行前固定：
   - 定义 `rMAD = 1.4826 × median(|x-median(x)|) / median(x)`；
   - fresh A 相对旧 `348.80 us` 的 drift 必须满足
     `|median(A)/348.80-1| <= max(3.0%, 3×rMAD(A))`；
   - probe identity B 相对 A 必须满足
     `|median(B)/median(A)-1| <= max(1.0%, 3×sqrt(rMAD(A)^2+rMAD(B)^2))`；
   - no-marker identity whole-event相对 production control须在 1.0% 内。
   Probe whole-event perturbation单独报告，不能直接充当 phase acceptance band；
6. B 在同一完整 launch 中执行原 P0–P2 predecessor与全部后续 phase，P3前无 host/device 额外
   触碰；L2/DRAM traffic 与 cache-hit evidence 不出现无法解释的初态差异；
7. correctness、work ledger、timing、resource、SASS 任一 gate失败即停止 R2 causal interpretation。

### C actual identity

C 必须锁定 actual graph 的：

- Prefix/Expand mangled+demangled symbols、template parameters、grid/block、PDL flags与 correlation；
- FlashInfer/CUTLASS source SHA、shared-library/cubin hash和 exact kernel entry resource record；
- BF16 input、NVFP4 output、block-scale config、fast-math/4-over-6 config及 per-expert scale mode；
- output correctness 和 fresh repeat spread。

自写 source-shape kernel、不同 template、不同 PDL predecessor 或不同 cubin 均不能通过 C gate。

## Correctness 与 work identity

每个 arm timing 前必须通过：

1. expert count与 route multiset exact match；
2. 每个输入 route occurrence 恰好生成一个 local routed record；主 fixture 的 unique-expert gate
   通过时可唯一配对，duplicate corner 只要求 occurrence multiset闭合；
3. `token/weight/expert` canonical metadata exact match；
4. packed FP4/scales 对各自 quant reference exact 或满足预先锁定的 dequantized tolerance；
5. tail expert、空 expert、重复 expert-id route（若 API允许）和 nonuniform hot-expert fixture；
6. 输出 sentinel/NaN overwrite 审查，不能只检查有限值；
7. static ledger 与 NCU dynamic load/store/atomic/instruction数量方向闭合；主 fixture 必须验证每个
   token 的 expert id 唯一，duplicate expert case 按 occurrence multiset 校验。

Timing fixture 使用 canonical M8192 case；corner fixtures只验证正确性，不混入主性能表。

## 测量、资源与 cache 协议

- 同一 5KP GPU UUID、application clock、GPU 无 foreign process；A/B/C 在同一资源租约与 session。
- 固定 container digest、Python、CUDA runtime/driver、CuteDSL、nvcc、ptxas、nvdisasm、NCU、NSys、
  FlashInfer source、CUTLASS submodule和 JIT cache root；manifest记录路径、版本与 hash。
- 记录每个 exact kernel entry 的 cubin hash、registers/thread、stack、static/dynamic SMEM、launch
  geometry、occupancy limit、compiler/JIT flags；不能只记录整个共享库 hash。
- 正确性先于 timing；固定 warmup/repeat/order/cooldown，保存 raw samples，不只保存 median。
- A/B 的 cache predecessor 固定为同 launch P0–P2；C使用 actual graph predecessor。另设一个预先
  声明的 cold-cache diagnostic 时，两边都执行同一 eviction policy，结果单列，不能事后挑 warm/cold。
- phase time、kernel duration、Prefix→Expand interval与 whole-operator event time使用各自原生分母，
  禁止相加或互换。

## NCU 与源码证据

只采回答机制所需的最小集合：

- Memory：DRAM/L2/global load/store bytes/requests、L1TEX global atomic sectors、local load/store与
  shared load/store；
- Compute：Executed warp instructions、FP/ALU/SFU/XU utilization、Issue active；
- Schedule：Achieved occupancy、Active/Eligible warps、Warp stalls（Wait / Long scoreboard /
  Short scoreboard / Barrier）与 Throttle；
- Resource：registers/thread、SMEM、stack、dynamic spill/refill；
- Source/Instruction：pair-head atomic、expert-row atomic、BF16 load、FP4 convert/pack、scale store和
  packed-row store 的 PC/SASS execution evidence。

A/B/D 的 NCU 都覆盖 whole fused launch，只能提供 operator-range可加计数与 source-PC证据；不能把
whole-launch utilization、WarpStateStats 或 stall 百分比归属 P3，也不能按 P3时间占比切分。P3
机制判定只接受：P3 marker time、未插桩 full-kernel event，以及 exact P3 PC 的 dynamic
instruction/atomic或 source-resolved stall sample；工具无法提供 PC attribution时该机制保持
`unresolved`。Whole-launch stall/utilization 只作 sanity。

另在 ledger 明确 ownership 差异：Fused 每 lane 处理一个 16-BF16 block；CUTLASS Expand 由两线程
各处理 8 values 共同形成一个 16-value block。该事实只描述 mapping，不能用跨-kernel总利用率百分比
直接归因 quant efficiency。C2 各百分比只在其独立 launch内解释；Prefix3 的 bytes/instructions可加，
occupancy/stall不加总。

## 判定树

1. **A fresh anchor不闭合**：先报告 source/compiler/cache drift；不运行或不解释 controls。
2. **B fidelity失败**：收口为“实验 clone 未保留 production P3 embedding”，列出失败的
   dataflow、resource或cache gate；不得用 B/C 比值解释 production。
3. **C actual identity失败**：C1/C2降级为 descriptive，不引用既有 `245.600 us` 作 fresh ratio。
4. **B通过且 `shared_equal_scale` 改善**：若 P3 与未插桩 e2e 同向改善、正确性和 resource gate
   通过，接受 shared-input composite benefit；没有 companion control 时不拆成纯量化或纯调度收益。
5. **`static_batch_schedule` 改善**，同时 exact P3 claim PC 的 dynamic atomic证据下降：支持
   persistent claim/scheduling 相关；没有 PC attribution时只记 timing现象。
6. **`precomputed_phys_row` 改善**，同时 row-allocation atomic与 dependent scoreboard/serialization
   证据下降，且 mapping load/traffic已计入：支持 dynamic row allocation/address dependency。
7. **controls均无明显改善，B的 quant/load/store exact-PC或memory traffic证据显示压力**：将重点转向
   FP4 quant、scale layout和GMEM效率，下一实验必须定位到 exact PC/data edge。
8. **C1与Fused P1+P2+P3差异明显，但B与C2接近**：差距属于 Prefix/map construction或阶段边界，
   不是 Q0/Pack 主体。
9. **latency、traffic、stall和correctness证据冲突**：结论标记 unresolved，只提出能区分冲突的
   下一最小实验；不用解释性文字填平冲突。

任何优化建议必须满足：

```text
observation → exact PC/data edge → mechanism → controlled counterfactual
            → uninstrumented benefit → correctness/resource regression gate
```

## 产物与收口

```text
exp_011_route_q0_pack_vs_chain/
  plan.md
  <full-kernel overlays/launcher/harness sources>
  tests/
  results/
    result.md
    manifest.json
    raw/
    derived/
```

- 所有脚本输出、trace、counter export和报告只放 `results/`；大体积 raw artifacts由 `.gitignore`
  排除，只提交可复核 summary、manifest与小型 derived tables。
- `result.md`以中文为主，按以下顺序展示：
  1. 对比约定与 fidelity verdict；
  2. A P3、C1 Prefix3+Expand、C2 Expand及B的时间；
  3. common logical work/traffic ledger；
  4. B/C与最小 controls的精简 NCU表；
  5. 已定位瓶颈、未闭合疑点和下一步。
- 报告动笔前执行 data-audit，对每个数字追溯 raw artifact；因果结论做一次轻量独立审计。
- 本计划只接受一次 Plan Review；review verdict与一次性修正摘要附在本文末尾，不发起第二轮。

## Plan Review

- 日期：2026-07-19
- Reviewer：subagent `/root/exp011_plan_review`
- Verdict：`✗ Misaligned`（已一次性修正；不进行第二轮 review）
- 已修正的重大 gaps：
  1. 取消 full-kernel `110×160` 与独立 `880×256` direct-row arm 的非法混用；
  2. 增加现成 `shared_equal_scale` composite control，并锁定适用范围、工作量与非纯量化归因；
  3. R3 改为 P1+P2+P3 与 Prefix3+Expand 的 raw-time stage context，禁止 ratio；
  4. correctness key 改为 occurrence multiset，不再假设 Fused 能恢复 topk slot；
  5. 禁止用 whole-launch stall/utilization 归因 P3，要求 exact-PC evidence或保持 unresolved；
  6. 将 CUTLASS exact public timing 与中间 tensor correctness harness 拆开；
  7. 在运行前固定 A-old、A/B identity 与 no-marker fidelity 阈值公式。
