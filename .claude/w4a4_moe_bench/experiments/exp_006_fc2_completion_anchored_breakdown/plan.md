# exp_006：FC2 GEMM 与 Atomic Scatter 边界验证

Status: **complete — ABI339 diagnostic estimate, Data Audit PASS**

## 1. Goal

验证 exp_004 中“atomic scatter 占 Fused whole-wall 超过 30%”是否由边界错分导致。
读者只看两个阶段：

```text
FC2 GEMM (A→D；含 epilogue / scale-cast / R2S / pre-scatter sync)
→ Atomic scatter (D→F；含 scatter loop / post-scatter sync)
```

内部 C/E marker 只用于边界求真，不作为读者级 phase。OMMA 是 warp-level 同步 MMA，
不得引入异步 issue/completion 语义。

exp_004 保留为 whole-kernel 诊断历史；本实验只 supersede 其 FC2 子阶段语义，不改写 exp_004 原始数据。

## 2. Locked Case and Identity

| Field | Locked value |
|---|---|
| Backend | `cutedsl_bf16_fused` |
| Shape | `M=8192, E=256, H=2048, I_tp=512, topk=8` |
| Launch | grid `(1,1,110)`；block `(160,1,1)`；1 CTA/SM |
| Warp roles | W0–W3 MMA consumers；W4 TMA producer |
| Tile | FC2 `M/N/K=128/128/128`；`epi_rest_m=1`；output tiles=16 |
| Task slicing | `_TASK_SLICE_CHUNK=1`；每 task 一个 K=128 FC2 partial |
| Production kernel SHA-256 | `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106` |
| Production dispatch SHA-256 | `cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4` |
| Container image digest | `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba` |

捕获必须记录 GPU UUID、driver、CUDA runtime、NVCC、Python package lock、FlashInfer commit、
CUTLASS commit、overlay/source/cubin/PTX/SASS hashes。任一 locked identity 不匹配即停止。

## 3. Arms and Measurement Protocol

两个 arm 必须由同一 production source、同一 builder 生成，并使用独立、全新的 JIT/output root：

1. `measurement_no_marker`：保留相同 overlay plumbing，但关闭 timestamp writes。
2. `completion_anchored_probe`：启用 `%globaltimer` task markers。

固定 fixture seed、workspace、route descriptor order、CUDA Graph 和 GPU clocks；每 arm warmup 2 次，
measured replay 5 次。捕获前检查无 foreign GPU process。两个 arm 都执行 reference correctness、
workspace/descriptor contract、fresh self-drift 和 latency；probe drift 以 cubin/resource/SASS 证据分级。

## 4. Event ABI and Phase Boundaries

每个有效 task 记录 339 个 event slots。基础 consumer event `0..6` 沿用 exp_004；`7..8` 是同一
task 内连续执行的 marker-pair calibration。16 个 FC2 output tile 各记录 20 个 event，tile `i` 的
base 为 `9 + 20*i`：

| Offset | Boundary |
|---:|---|
| 0..3 | `A0..A3`：对应 compute warp 的 `consumer_try_wait` 前 lane0 edge |
| 4..7 | `C0..C3`：对应 compute warp 最后一条 OMMA 后、首次 epilogue 指令前 |
| 8..11 | `D0..D3`：对应 compute warp 完成 scale/cast、R→S、fence 并通过 pre-scatter barrier 后 |
| 12..15 | `E0..E3`：对应 compute warp 的 atomic scatter loop 完成、post-scatter barrier 到达前 |
| 16..19 | `F0..F3`：对应 compute warp 通过 post-scatter barrier 后的 lane0 edge |

W4 的 10 个 producer event 使用 `329..338`。锁定 case 必须静态/运行时断言 `epi_rest_m=1` 与
`_TASK_SLICE_CHUNK=1`，否则现有 ABI 不合法并立即停止。

Additive FC2 读者阶段：

- `FC2 GEMM = Σ(max(Ai)→min(Di))`：pipeline wait/load、accumulator clear、OMMA、epilogue、
  scale/cast、R→S 与 pre-scatter sync。
- `Atomic scatter = Σ(min(Di)→max(Fi))`：scatter loop 与 post-scatter sync。

C 与 E 仅保留为 marker 级证据，用于确认 D 边界前后各自包含什么；不对它们单独命名为
GEMM completion 阶段。两个读者阶段互斥，也不是四个 warp service time 的相加。
每个 warp 必须满足 `Ai≤Ci≤Di≤Ei≤Fi`，并额外验证 `max(Ci)≤min(Di)`；保留全部 per-warp raw edge。

实现期 eager hard gate 证明单一 W0 `F` 不是可靠的 cross-warp timer edge：生成的 PTX/SASS 均保持
`Ei CS2R/store → BAR.SYNC.DEFER_BLOCKING → W0 F CS2R/store`，但 raw capture 仍出现 512 个
`W0 F < max(Ei)`，最大反转 480 ns，且 timestamp 以 32 ns 量化。因此 ABI 不加入容差，也不放宽
fail-closed gate；改为同 warp `Ei→Fi` 配对，再以 `max(Fi)` 定义 collective completion。

后续 forward-test 又触发同一 fail-closed 原则：W0-only `D` 虽能证明 GEMM materialization 已在 scatter
之前完成，却不能代表 W1–W3 的 scatter 起点；同时 A/C 也不应由代表 warp 替代。最终 ABI 因此扩展为
逐 warp `Ai/Ci/Di/Ei/Fi`，以 `max(Ai)→max(Ci)→min(Di)→max(Ei)→max(Fi)` 形成声明过的 collective
critical-path timeline。旧 195-slot 数据保留为 superseded attempt，不用于最终 phase 数字。

## 5. Boundary Proof and Hard Gates

捕获 exact probe cubin 后，必须用 SASS/source mapping 证明：

1. `Ai` 的 marker store 紧邻 consumer TRYWAIT 之前。
2. `Ci` 位于对应 warp 最后一条 OMMA 后、首次 epilogue 指令前。
3. `Di` 位于对应 warp 完成 scale/cast/R→S 以及 pre-scatter barrier 后。
4. 每个 same-warp `Di→Ei` 内不存在 OMMA、epilogue 或 R→S 指令；仅允许 scatter path。
5. `E0..E3` 分别由对应 compute warp 的 lane0 写在 scatter loop 后、post-scatter barrier 前；
   `F0..F3` 分别由同一 compute warp 的 lane0 写在该 barrier 后。

任一证明失败，禁止使用 `FC2 GEMM`、`Atomic scatter` 边界语义，实验结论标记 inconclusive。

每个 replay 还必须通过：event exact-fill、unused sentinel、task/CTA monotonicity、逐 warp 跨 tile
`Fi≤A(i+1)`、task envelope、
无 slot overwrite、5/5 correctness、CUDA Event 交叉检查。保存每个 task 的 `valid_rows`、expert、M tile、
slice index 与 descriptor-order hash，允许直接计算 `valid_rows ↔ scatter body` 关系。

## 6. Analysis and Decision

报告 5 replay 的 mean/p50/p95/CV、SM-equivalent wall share、FC2 内部 share、每 task/tile distribution，
并对 `valid_rows` 与 CTA scatter body 做直接回归；warp 级解释必须使用各 warp 实际行数
`clamp(valid_rows - warp_m_base, 0, 64)`，不得把 W0 时间直接回归到全部 `valid_rows`。phase sum 与
相同 task envelope 必须精确闭合；
residual 必须命名且不能作为边界正确性的证据。

Phase time/share 发布前必须生成结构化 ownership/boundary-data audit：逐 phase 绑定 semantic owner、
start/end data state、downstream-ready handoff、included/excluded work、fine-marker aggregation、production +
probe-overlay source、loaded SASS、timing authority/unit/denominator。所有 share 从 5 replay × 110 CTA 的
`%globaltimer` SM-equivalent denominator 重算，不得使用 CUDA-event median 代替；audit/hash 未进入
manifest 时保持 blocked。

跨 arm 只比较完整 fused launch latency 与 resource drift；不比较 control 与 probe 的内部 phase。
本实验不产生新的 fused/chain speedup，只提供经边界验证的 FC2 内部 breakdown。回填 exp_002 时，
不得对 A→C、C→D 或 D→F marker 区间单独配对 chain kernel 并声称 speedup；
既有横向表只有在 semantic start 与 completion endpoint 同时相同的合并区间上才能计算对比；本实验
没有证明这样的 FC2 共同起点，因此 relative delta/speedup 必须留空。

Marker/drift 决策门禁预注册如下，不做事后选择：

- calibration 使用每 task 连续 `7→8` marker 的 median delta；若该成本超过某子阶段 mean 的 10%，
  该子阶段只报告原始上界并标 inconclusive，不做校正后点估计。
- probe/control median whole-launch overhead `≤5%`、occupancy 不变且 STACK/LDL/STL static count drift
  均 `≤25%`：允许报告 diagnostic estimate。
- overhead `(5%,10%]` 或任一 static local-resource drift `(25%,50%]`：只报告上界/范围。
- overhead `>10%`、achieved occupancy 改变，或任一 static local-resource drift `>50%`：停止 phase 归因。
- aggregate phase share 的 5-replay CV 必须 `≤5%`；否则该 phase 标 inconclusive。

结果写入本实验 `results/`。通过 Data Audit 后，回填 exp_002 的 FC2 表格与边界说明；exp_004 只增加
指向新证据的 supersession 说明（若确有必要），不得覆盖其 raw/derived/canonical artifacts。

## 7. Stop Conditions

- locked source/environment/shape/launch 不匹配；
- correctness、workspace、descriptor、event ABI 或 SASS boundary proof 失败；
- probe overhead/resource drift 大到使 phase 分布无法解释；
- task descriptors 未被保留，导致 scatter 工作量无法自证。

任何 stop condition 触发时，保留证据并报告 blocked/inconclusive，不用推测补齐结论。

## Plan Review

Verdict: **GO_WITH_FIXES**

独立 review 指出的五项风险已在本版一次性修正：用 `E0..E3` 分离 scatter 慢尾与 post-sync；
按 warp 实际行数解释 `valid_rows`；加入 marker-pair calibration；把 probe drift/CV 变成预注册数值门禁；
删除本实验产生新 fused/chain speedup 的范围。实现期 hard-gate 触发后的 `F0..F3` ABI 修正记录在
§4，属于同一 reviewed experiment 的 fail-closed instrumentation correction，不启动第二轮 Plan Review。
后续 `D0..D3` forward-test 修正同样只修复既有 pre-scatter boundary hard gate，不改变 case、问题、arm、
测量协议或判定门禁，因此按 single-round-per-exp 规则不启动第二轮 Plan Review。
