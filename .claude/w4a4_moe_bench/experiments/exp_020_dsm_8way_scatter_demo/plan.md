# exp_020：4-CTA DSM 合并 Scatter Demo

## 状态

**Closed / Reject。** Demo correctness 通过，但 DSM-8 在 AB、BA 两种顺序下都稳定回退约 91.29%；
因此不进入 Opt 集成实验，也不追加 NCU/phase 深挖。结果见 [results/result.md](results/result.md)。

## 目标

在不修改 production 和 `moe_dynamic_kernel_opt.py` 的前提下，验证一个具体机制：

```text
当前：4 个 FC2 K-slice × topk 8 → 每个输出元素 32 次 GMEM REDG
候选：4 个 slice 在 4-CTA cluster 的 DSM 中先合并 → 每个输出元素 8 次 GMEM REDG
```

回答两件事：

1. SM120 上的 4-CTA cluster + DSM merge 能否保持正确、无 spill，并确实把 REDG work 降低 4 倍；
2. DSM remote load、FP32 merge 和两次 cluster barrier 的全部成本计入后，是否仍比 Direct-32 更快。

Demo 通过只代表“允许集成验证”，不直接证明完整 Opt 会加速。Demo 回退则标记 Reject，不改 Opt。

## 已锁定事实

- 目标 GPU 为 SM120 5KP；硬件与 CUDA runtime 均支持 Thread Block Cluster/DSM，正式运行仍先做
  cluster-launch 与 occupancy preflight。
- 当前 Opt 每个 task 只处理一个 FC2 `K=128` slice；4 个 slice 对同一 routed row 分别 Scatter。
- 当前 Scatter 为 8 个 math warp、每 warp `32×64` strip、每次一条 BF16x8 REDG。
- 当前对比中的 SGLang FP8 output reduce 从 GMEM intermediate 读入并在寄存器做 topk=8 reduce；它没有
  使用 DSM。DSM 是本实验的新设计，不是复刻 Triton/SGLang 实现。

## 两臂与唯一变量

两臂都用 CuTeDSL、同一 cubin build environment、同一 `4-CTA cluster`、每 CTA 288 threads
（8 math warp + 1 idle/control warp）、同一真实 route/task metadata、相同 BF16 `sC` layout、相同
SMEM footprint 和相同输出清零策略。

### A. Direct-32 baseline

- CTA rank 0..3 分别代表 FC2 slice 0..3；
- 每个 CTA 从自己的 local `sC` 读取 BF16 partial，转 FP32、乘 route weight、pack 后直接 REDG；
- 4 个 slice 与 topk=8 共同形成每个最终输出元素 32 次 global reduction；
- 不人为加入 Candidate 才需要的 cluster barrier。

### B. DSM-8 candidate

- 4 个 CTA 先把各自 BF16 partial 写入相同地址布局的 local `sC`；
- **每个 N128 output tile** 都由全部线程依次执行 async-shared proxy fence 与 publication
  cluster barrier；
- 四个 CTA 共同承担互斥的输出区域，每个 owner 通过 remote SMEM pointer 按固定 slice 0→3 顺序
  读取并 FP32 求和，再乘一次 route weight、pack，并只发出一次 REDG；不能退化为 CTA0-only reducer；
- **每个 output tile** 都由全部线程执行 consumed cluster barrier 后才复用 `sC`；最后一 tile
  也不能提前退出，保证 DSM 生命周期合法。

唯一机制差异是 `4 次 global REDG` 与 `4-way DSM merge + 1 次 global REDG` 的替换。权威 timed DAG
锁定为：

```text
kernel launch
  → 每 tile 相同的 metadata staging + deterministic BF16 sC generation
  → local completion/sync
  → Direct Scatter，或 DSM publication/merge/Scatter/consumed sync
  → 下一 tile
  → kernel exit
```

Output clear 与固定 L2 eviction 在 CUDA event 外；kernel 内共同 staging/generation 在 event 内。
Generation 使用同一条轻量确定性 BF16 公式和 production `sC` layout，不从 GMEM partial 读取。唯一性能
接受依据是未插桩 whole-demo CUDA-event wall；Scatter phase probe 只做方向与机制交叉检查。

## Production-like 工作量

主 case 固定 `M=8192, E=256, H=2048, I_tp=512, topk=8, tile=128×128`：

- fresh current-Opt canonical fixture；若复用 exp_010 fixture，必须严格校验 input/routing/task/source hash；
- 65,536 个有效 routed rows、637 个 `(expert, M-tile)` logical groups、2,548 个 slice tasks；
- 每个 logical group 对应一个 4-CTA cluster，rank 0..3 的 slice 集合必须恰为 `{0,1,2,3}`，
  `valid_rows` 必须一致；四个 rank 的每一行还必须具有相同 `(expert, M-tile, token_id,
  route_weight)` 顺序；每个 group 依次处理相同的 16 个 N128 output tiles；
- 保留实际 token collision、route weight、tail 和 padded-row 分布，不使用 unique destination 或均匀无冲突数据。

工作台账：

| Arm | BF16x8 REDG | reduction payload | 每个 `[token,h]` 的 contribution |
|---|---:|---:|---:|
| Direct-32 | 67,108,864 | 1 GiB | 32 |
| DSM-8 | 16,777,216 | 256 MiB | 8 |

DSM-8 额外读取约 1 GiB BF16 partial（约 256 MiB local + 768 MiB remote DSM），并执行约
402,653,184 次 FP32 add；这些成本不得移出计时边界。

## 正确性与身份门禁

1. Runtime preflight：SM120、cluster launch supported、cluster size=4 可启动，并记录 exact
   block/cluster/SMEM 下的 max active clusters。
2. Fixture：每组四个 slice exactly once、`valid_rows` 一致、每 token 恰有 8 条 route；对
   `(expert, M-tile, output-tile, physical-row, token_id, route_weight)` 做逐字段与 hash 对齐，并加入
   slice-coded canary，识别漏 slice、重复 slice、错行或错 owner。
3. Ownership：Direct 每个有效 partial 恰发出一次；DSM 每个 route/output vector 只有一个 spatial owner，
   四个 CTA 均有有效 merge work；覆盖 `valid_rows=1/31/32/33/63/64/65/95/96/97/127/128`。
4. Numeric：保留每 slice 的 FP32→BF16 epilogue boundary；只锁定 DSM 内 slice 0→3 的 FP32
   求和顺序。两臂都对照高精度语义 oracle 和预先固定的误差界，并做多次稳定性检查；DSM 另对现有
   完整 MoE tolerance/finite/sentinel 做校验。由于跨 expert/top-k 的 BF16 atomic 到达顺序不确定，
   不把固定 atomic rounding 顺序或两臂 bitwise 相等作为合法门禁。
5. Binary：记录 source/JIT/cubin hash、CUDA/nvcc/CuTeDSL/GPU UUID、grid/block/cluster、register/SMEM/
   stack；SASS/NCU 证明存在 DSM remote access 与 cluster barrier、没有 GMEM partial fallback。
6. Resource：static 与 dynamic evidence 均须 zero spill；不能用 standalone 较低 register footprint 冒充
   production resource identity，Demo 结论只覆盖 reduction mechanism。
7. DSM protocol：每个 cluster 动态确认 16 次 publication + 16 次 consumed barrier；288 个线程全部
   收敛参与。每 tile 的顺序固定为 local epilogue completion → `fence.proxy.async.shared` 等价 fence
   → publication barrier → remote load/merge → consumed barrier → `sC` reuse/exit。

## 性能测量

- 同一空闲 5KP、锁定 2377 MHz、同一进程；每 arm 先 warmup 20 次。每次 output clear 后执行同一
  固定 L2 eviction，再记录 start event，保证 clear/eviction 不进入 timed DAG。
- uninstrumented CUDA-event kernel wall 为唯一性能接受依据；单进程做 5 个 paired groups，按
  `AB/BA/AB/BA/AB` 交替首发，每组每 arm 100 个 launch。保存全部 raw samples、每组 arm median、
  paired median speedup 与 delta；CV 定义为每组每 arm 的 `stdev/mean`。
- 对 5 个 group-level paired speedup 做固定 seed bootstrap，要求 one-sided 95% lower bound > 0；
  同时记录每组开始/结束的实际 clock、温度与功耗，发生 foreign process 或 clock drift 即丢弃整组重测。
- diagnostic phase probe 从 local `sC`/metadata ready 后开始，到最后一个 DSM consumed barrier 或
  Direct Scatter 完成后结束；probe 只解释机制，不能覆盖 uninstrumented wall 判定。
- 仅在 correctness 通过后各采一次最小 NCU/SASS：REDG work、DSM/shared/global traffic、barrier/stall、
  occupancy、register/SMEM/stack/spill。禁止先跑完整 deep profile。

## 判定

- **Accept / 允许集成**：全部 correctness/identity/resource gate 通过；REDG 动态工作量精确 4:1；
  5 个 paired group 全部为正收益，group-level whole-demo median speedup ≥2%、bootstrap 95% lower
  bound > 0，且所有有效 group 的 per-arm CV ≤1.5%。Phase probe 若反向则降级为 Unresolved，不能 Accept。
- **Reject**：任一 correctness/DSM/zero-spill gate 失败，或 whole-demo/phase 没有可重复正收益。
- **Unresolved**：计时正收益但 REDG/DSM 机制证据不闭合，或 phase 与 whole-demo 方向冲突；只补能区分
  冲突的最小证据，不集成 Opt。

Accept 后另开集成实验：先生成独立 Opt overlay，重新验证 scheduler/grouping、完整 M=256..8192
correctness、zero spill 和未插桩 E2E；通过后才提升为 accepted Opt。Reject 时不产生任何 Opt 改动。

## 产物

```text
exp_020_dsm_8way_scatter_demo/
  plan.md
  dsm_scatter_demo.py
  run_demo.py
  run_remote.sh
  results/
    result.md
    manifest.json
    raw/
```

保持一个 kernel source、一个 harness、一个 remote launcher；不为每个 arm 重写脚本，不保存编译缓存、
临时 dump 或重复采样产物到 git。

## Plan Review

- 日期：2026-07-21
- Reviewer：subagent `/root/exp020_plan_review`
- Verdict：`✗ Misaligned`（以下缺口已一次性修正；按 single-round 规则不再复审）
- 已修正的重大 gaps：
  1. 锁定唯一权威 timed DAG，统一 output clear、L2 eviction、metadata 与 partial generation 的边界；
  2. 将 proxy fence、publication/consumed barrier 和 288-thread convergence 明确到每个 output tile；
  3. 将四个 slice 的合法合并从 descriptor-level 提升到逐行 route/weight/output-tile identity；
  4. 取消错误的 deterministic atomic oracle，改用高精度语义误差界和多次稳定性；
  5. 固定 warmup/repeat、paired 统计量、CV 对象、bootstrap lower bound 与 clock/temperature/power gate。
