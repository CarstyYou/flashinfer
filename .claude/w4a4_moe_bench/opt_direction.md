# Optimization Directions

状态：`[todo]` 待验证，`[done]` 已接受，`[reject]` 已拒绝。

## 当前优化方向

### [done] 1. 提高整个 kernel 的有效并行度

- 目标不是只把 CTA 配置成 8 个 math warp，而是让各 phase 都有效利用 W0–W7。
- 这是总方向，不作为一次大改版实验；Scatter 与 Route/Q0 分成两个独立子方向逐步实现。
- FC1、SwiGLU/Q1 和 FC2 已使用 8 个 math warp；重点补齐两个 memory-bound phase：
  - **[done] Scatter / exp_014**：W0–W7 已无重漏地分担 R2S 后的 Scatter；M2048/4096/8192
    fused E2E 分别提升 7.59%/5.69%/7.43%，static/dynamic zero-spill。
  - **[done] Route/Q0 / exp_016**：W0–W8 从 pair-major 改成 token-major；同一 token 的 BF16
    input load 与 block absmax 复用给 top-8 routes，量化和输出数量保持不变。P3 grid critical wall
    降低 58.25%；M256–8192 fused E2E 全部提升，M8192 提升 9.71%，static/dynamic zero-spill。
    此实现仅锁定 `topk=8 + [E] scale + full_tile_publish=0`，不是通用 top-k 路径。
- 每次只修改一个 phase，用 correctness、spill、phase latency、memory throughput、warp activity 和 E2E
  判断并行度是否真正提高。

### [todo] 2. 安全地加深 pipeline stage

- 当前 Stage2 偏浅；直接把整个 kernel 改为 Stage4 需要 `131,072 B` SMEM，超过 SM120 的
  `101,376 B` 上限，该直接方案已 `[reject]`。
- 后续尝试生命周期互斥 buffer 的安全复用，或 A/SFA 与 B/SFB 使用非对称 stage。
- 不采用已经 reject 的 compact epilogue 来换取 SMEM。

### [todo] 3. 跨 worktile 的 phase overlap

- 当前 worktile 基本按 `Route/Q0 → FC1 → SwiGLU/Q1 → FC2 → Scatter` 串行推进。
- 优先尝试 `Scatter(i) ∥ Claim/Route/Prologue(i+1)`，再评估是否扩展到部分 `Q0(i+1)`。
- 利用 Scatter 阶段空闲的 W4–W7，并通过独立或双缓冲状态管理跨 tile 的数据与 barrier。
- 需要确认 atomic 正确性，以及 Scatter 与 Route/Q0 对 CUDA Core、LSU、register 和 SMEM 的争用。

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

### [reject] Intern Stage4 compact 实现

- M4096/M8192 correctness 失败，不能作为优化实现。
- 只保留其 SMEM 生命周期复用机制作为方向 3 的参考。

## 验证与接受条件

`单一优化问题 → 最小候选 → correctness → static/dynamic zero-spill → E2E → accept/reject`

- 不修改 Production 或 Intern kernel；候选从当前 accepted opt baseline 独立派生。
- 快速验证不建 exp；值得保留的调查再建立正式 exp 与证据目录。
- Reject 的候选不进入 `moe_dynamic_kernel_opt.py`。
