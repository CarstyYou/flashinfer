# Optimization Directions

状态：`[todo]` 待验证，`[done]` 已接受，`[reject]` 已拒绝。

## exp_017 Takeaway

- M8192 的未插桩 E2E 距离 2× Triton FP8 目标仍差 `248.032 µs`。诊断 phase 投影中，
  `Route + Q0 + Pack` 为 `94.038 µs / 7.06%`，`Atomic Scatter` 为
  `371.964 µs / 27.91%`；二者合计占 `466.002 µs / 34.97%`，因此当前优化重心应从
  GEMM/spill 转向输入量化与输出归并边界。
- exp_014/exp_016 证明了 8-warp Scatter 和 token-major Route/Q0 是有效改法，但 `[done]`
  仅表示这轮策略已接受，不表示对应 phase 已经优化完。
- 优先级锁定为：**P0 输出归并 → P1 Route/Q0 residual → 条件触发 Q1**。
- Q1 当前只有 `33.859 µs / 2.54%`，且与 Triton 的 phase 边界不完全等价；目前只能作为候选，
  不能写成已确认瓶颈。只有 P0/P1 收益不足，或发现 Q1 阻塞 phase overlap / TC cadence 时再深挖。
- phase 时间用于确定调查优先级，不是可直接回收的 latency budget；跨 backend 的非等价 phase
  也不能据此声明正式 speedup。

## exp_019 Opt vs Eric Takeaway

- Latest Opt 继续作为主干。M8192 时 Eric 慢 `352.340 µs / 26.35%`；同边界 phase 中主要回退是
  Route/Q0 `+240.941 µs`、Scatter `+92.449 µs` 和 FC2 `+56.399 µs`。
- 保留 Opt 的 token-major Route/Q0、8-warp Scatter、paired N64 与 zero-spill。Eric production cubin
  有 `947,200 / 509,600` 次动态 spill load/store，不能整体合并。
- Eric 的 FC1 Gate+Up+SwiGLU bundle 在 M8192 快 `23.513 µs / 5.29%`，说明更深 pipeline / buffer
  schedule 值得做单变量实验；Stage4、4-warp、N128 与 compact epilogue 同时变化，当前不能把收益归因
  给任一项。
- M1024 正式 benchmark 中 Eric 快 `21.348 µs`，但 fresh diagnostic control 只快 `7.168 µs`；方向一致、
  幅度差 `2.98×`，本轮不对 crossover 做 phase 因果归因，也不为此启动 deep NCU。

## 当前优化方向

### [done] 1. 首轮 phase 并行化

- 目标不是只把 CTA 配置成 8 个 math warp，而是让各 phase 都有效利用 W0–W7。
- 这是总方向，不作为一次大改版实验；Scatter 与 Route/Q0 已按两个独立子方向逐步实现。
- FC1、SwiGLU/Q1 和 FC2 已使用 8 个 math warp；首轮也已补齐两个 memory-bound phase：
  - **[done] Scatter / exp_014**：W0–W7 已无重漏地分担 R2S 后的 Scatter；M2048/4096/8192
    fused E2E 分别提升 7.59%/5.69%/7.43%，static/dynamic zero-spill。
  - **[done] Route/Q0 / exp_016**：W0–W8 从 pair-major 改成 token-major；同一 token 的 BF16
    input load 与 block absmax 复用给 top-8 routes，量化和输出数量保持不变。P3 grid critical wall
    降低 58.25%；M256–8192 fused E2E 全部提升，M8192 提升 9.71%，static/dynamic zero-spill。
    此实现仅锁定 `topk=8 + [E] scale + full_tile_publish=0`，不是通用 top-k 路径。
- 每次只修改一个 phase，用 correctness、spill、phase latency、memory throughput、warp activity 和 E2E
  判断并行度是否真正提高。

### [todo] 2. 输出归并 residual 定位与优化

- 当前 P0：即使 exp_014 已让 W0–W7 共同执行 Scatter，诊断投影仍为
  `371.964 µs / 27.91%`，不能把 8-warp mapping 已完成误写为 Scatter 已收口。
- 先用 matched、低扰动 probe 区分 `SMEM LDS → scale/pack → REDG → post-sync`，判断 dominant edge；
  现有证据只定位到这些语义区间，尚未证明 barrier、REDG completion 或 work imbalance 中谁是根因。
- 只有定位问题点后才设计单变量候选；以 correctness、zero-spill、Scatter phase 与未插桩 E2E
  同时改善作为接受条件。

### [todo] 3. Route/Q0 residual 定位与优化

- 当前 P1：token-major 已接受，但合并边界 `Route + Q0 + Pack` 仍为
  `94.038 µs / 7.06%`。
- 在保持 exp_016 work mapping 与 logical work 不变的前提下，先做低扰动 phase-local 细分，找到
  routing、quantization、pack/store 中的 dominant path，再决定优化对象。
- 该合并边界不能与 Triton 的 Q0 kernel 直接计算 phase speedup。

### [todo] 4. 安全地加深 pipeline stage

- 当前 Stage2 偏浅；直接把整个 kernel 改为 Stage4 需要 `131,072 B` SMEM，超过 SM120 的
  `101,376 B` 上限，该直接方案已 `[reject]`。
- 后续尝试生命周期互斥 buffer 的安全复用，或 A/SFA 与 B/SFB 使用非对称 stage。
- 不采用已经 reject 的 compact epilogue 来换取 SMEM。
- exp_019 提供新的启动依据：Eric 的整个 FC1 bundle 快 `23.513 µs`。下一候选只改变 Opt 的 FC1
  pipeline / buffer schedule，保持 8 math warps、paired N64、M128 epilogue 与其他 phase 不变；
  compiler/dynamic zero-spill 是硬门禁。该结果不能预先记为 Stage4 收益。

### [todo] 5. 跨 worktile 的 phase overlap

- 当前 worktile 基本按 `Route/Q0 → FC1 → SwiGLU/Q1 → FC2 → Scatter` 串行推进。
- 优先尝试 `Scatter(i) ∥ Claim/Route/Prologue(i+1)`，再评估是否扩展到部分 `Q0(i+1)`。
- exp_014 后 Scatter 已使用 W0–W7，不能再假设 W4–W7 空闲；需要验证跨 tile 的角色分时、独立
  producer 或流水重排，并通过独立/双缓冲状态管理数据与 barrier。
- 需要确认 atomic 正确性，以及 Scatter 与 Route/Q0 对 CUDA Core、LSU、register 和 SMEM 的争用。

### [todo] 6. Q1 residual（条件触发）

- Q1 诊断投影为 `33.859 µs / 2.54%`，优先级低于 Scatter 与 Route/Q0。
- 仅在 P0/P1 不足以接近目标，或证据表明 Q1 阻塞跨 phase overlap / TC cadence 时启动调查；
  第一轮只定位边界与 cadence，不预设优化方案。

## 已完成或拒绝的尝试

### [done] exp_008：双 N64 FC1 与 A/SFA 复用

- 当前 accepted baseline：`moe_dynamic_kernel_opt.py`。
- CTA 使用 8 个 math warp + 1 个 DMA warp；FC1 使用双 N64 Gate/Up，FC2 保持 N128。
- correctness、static spill、dynamic spill 和 E2E 已通过。

### [done] exp_014：Scatter 8-warp work mapping

- Scatter 从 W0–W3 的 `64×64/warp` 改为 W0–W7 的 `32×64/warp`。
- 主要 prefill case 稳定提升，已进入 `moe_dynamic_kernel_opt.py`。

### [done] exp_016：Route/Q0 token-major 输入复用

- W0–W8 每轮各处理一个 token，复用该 token 的 BF16 load 与 block absmax，再分别完成 top-8
  per-expert quant/store。
- 组合机制同时减少 producer claims 并改变 route metadata ownership；收益不拆归给单一子变化。
- 正确性、P3 phase、静态/动态 spill 与完整 E2E sweep 均通过，进入
  `moe_dynamic_kernel_opt.py` 的锁定 topk-8 实验路径。

### [reject] exp_013：compact epilogue

- correctness 通过，但产生 `24 B/thread` stack traffic。
- M256/M8192 分别回退 0.603%/0.806%。

### [reject] Eric Stage4 compact 作为主优化路径

- 修正版在 exp_018 的六个 M 上 correctness 全部通过；但相对 Latest Opt，M2048/4096/8192
  分别慢 4.58%/15.47%/20.85%，因此不并入当前主优化 kernel。
- 只保留其 SMEM 生命周期复用机制作为方向 4 的参考。

## 验证与接受条件

`单一优化问题 → 最小候选 → correctness → static/dynamic zero-spill → E2E → accept/reject`

- 不修改 Production 或 Eric kernel；候选从当前 accepted opt baseline 独立派生。
- 快速验证不建 exp；值得保留的调查再建立正式 exp 与证据目录。
- Reject 的候选不进入 `moe_dynamic_kernel_opt.py`。
