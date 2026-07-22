# exp_026：RTX 5KP Hardware Ceiling Calibration

## 目标

按 KDK `calibrate-gpu-ceilings` 的 `development` 口径，在当前 W4A4 MoE
目标 RTX 5KP 上建立最小、可复用的 calibrated hardware profile，并用于重算：

- exp_024 CUTLASS NVFP4 chain 的硬件 ceiling efficiency；
- exp_025 SGLang Triton FP8 chain 的硬件 ceiling efficiency。

读者报告以百分比为主；原始 `Ops/clk/SM`、TFLOP/s、GB/s、cycles 与推导输入
保存在 profile/model 中。这个实验只建立 hardware roof，不宣称 operator SOTA。

## 目标硬件与执行身份

- Host：`xiy@10.6.142.16`，期望 hostname `R6KD-CX8aaS-GPU-16`。
- GPU：`GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`，SM120，期望 110 SM，
  application graphics clock 2377 MHz。
- GPU 必须无 compute process、memory used `<=256 MiB`，并取得
  `/tmp/kdk-direct-ssh-gpu-leases/R6KD-CX8aaS-GPU-16_<UUID>` 独占 lease；不触碰
  GPU 0/1 上正在运行的 SGLang/VLLM workload。
- Container：`nvcr.io/nvidia/pytorch:26.05-py3`，image ID
  `sha256:a4e056e1d34a5cc9387512ffa3abeed778e3dc7966633c5154d771705d8835ac`，
  digest `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`。
- Canonical benchmark：<https://gitlab-master.nvidia.com/xiy/sm120_mma_benchmarks>，
  base commit `1515eaa316436065f6c44471994d24428c021180`。先在 owning repo 增加
  exact E4M3×E4M3 no-scale 与 exact NVFP4/full-card saturator mode；capture 前冻结
  新 clean commit、remote、source/binary SHA256，禁止用 base commit 的缺失 mode。
- Build：CUDA 13.2 container 内编译 `sm_120a`；保存 driver、runtime、nvcc、
  CMake、CUTLASS dependency、binary SHA256 与完整命令。

## 最小校准集

### 1. Tensor Core per-cycle ceiling

运行 `SM120_benchmark` 5 次，保留全部输出，分别提取：

- `NVFP4_FP32_OMMA16864_VS16`：exp_024 的 exact instruction/scale mode；
- `E4M3_E4M3_FP32_QMMA16832`：exp_017 SASS
  `QMMA.16832.F32.E4M3.E4M3` 的 exact no-scale record；当前 base commit 不存在，
  必须先新增并用 SASS 验证；
- `E4M3_E4M3_FP32_QMMA16832_VS32`：block-scaled FP8 候选。

每个值必须核对实际 SASS/dispatch 后才能作为 denominator；不允许因为数值更有利
而选择。profile 使用 5 次
`Ops/clk/SM` 的 median，并保存 min/max/CV。该测试只测 per-cycle capability，
命名为 `per_sm_instruction_ceiling`；它的单 warp × 四 subpartition 外推不能命名为
full-card sustained peak。

### 2. Exact-mode full-card compute window

在 canonical `SM120_back_to_back_throttle_benchmark` 中增加显式 mode，并保留默认
行为兼容：

- `nvfp4-e2m1-vs16`：OMMA 16×8×64、UE4M3 scale；
- `fp8-e4m3-noscale`：QMMA 16×8×32，无 scale operand；
- `fp8-e4m3-vs32`：现有 block-scaled diagnostic。

先对每种 mode 运行 `blocks/SM = 4/8/12/16`、
`8 kernels × testtimes=16000 × gap=0` 的 saturation sweep，并记录 compiler
register count 与 occupancy API 给出的最大 active one-warp blocks/SM。选择达到该
mode 最大中位吞吐 99% 的最小 grid density；sweep 统计排除 rested kernel 0，
未达到 plateau 就不能称 calibrated roof。

每种 exact mode 再使用选定 grid density，运行
`32 kernels × testtimes=64000 × gap=0`，各 3 次；每次已有 3 次 warmup 和 2 秒
cooldown。输出每个 kernel 的 fixed FLOPs、duration 与 TFLOP/s。`kernel 0` 仅表示
rested first observation；最后 8 个 kernel 的 median 表示该约 100 ms 序列内的
`sustained-window` roof，不泛化为无限时长 thermal steady state。

另对每种 mode 运行 `16 × 2000 × 50000 us gap` 一次，只验证恢复/boost 方向。
保存 20 ms clock/power/temperature telemetry。full-card percentage 只消费 exact mode
和相同 window 的记录。

### 3. DRAM sustained ceiling

运行 `SM120_DRAM_bw_benchmark` 3 次。每次内部为 512 MiB buffer、warmup、
10 trials、每 trial 4 passes；profile 对三次输出的 read/write/copy/D2D median
再取 median，并保存变异度。各方向分别登记，不合并成一个 DRAM 数字。

对 streaming read/write/copy 的连续 4 passes 做一次最小 NCU physical DRAM
read/write byte 校验，证明 benchmark payload 与 target NCU hierarchy scope 的关系；D2D 只作 copy-engine
reference，不参与 kernel roof。对任意目标读写量 `R/W`：

```text
T_dram_floor = max(R / BW_read, W / BW_write, (R + W) / BW_copy)
DRAM resource efficiency = T_dram_floor / T_target
```

只有 target 与 calibration 都有兼容 physical DRAM bytes 时才计算。Reduction/atomic
仍只能称 `DRAM resource efficiency`，不是完整 op ceiling。

## 推导与百分比

```text
per-cycle calibration ratio = measured work/clk/SM / official work/clk/SM

target-window calibrated roof =
    measured work/clk/SM * installed 110 SMs * target-window effective clock

hardware ceiling efficiency =
    target achieved work rate / compatible calibrated roof

cycle-normalized executed efficiency =
    target executed work /
    (measured work/clk/SM * target sm__cycles_elapsed.sum)

cycle-normalized useful efficiency =
    target useful work /
    (measured work/clk/SM * target sm__cycles_elapsed.sum)

padding efficiency = target useful work / target executed work
```

`sm__cycles_elapsed.sum` 必须来自目标同 launch NCU，且核对其聚合已经覆盖所有
110 SM；不把少量 active SM 当作分母以隐藏并行度损失。优先使用 cycle-normalized
efficiency；只有目标证据没有兼容 elapsed-cycle counter 时，才用与 target 相同窗口的
exact-mode full-card roof。
Boost、sustained 与固定 application-clock 结果分开，不混成一个百分比。

对于 memory/transform op：只有 physical traffic scope 与 calibrated read/write/copy
scope 对齐时才计算百分比。Logical payload 不能除以 DRAM roof。Reduction/atomic
缺少匹配 microbenchmark 时完整 op ceiling 仍标 unavailable，只允许显示具名的
`DRAM resource efficiency`。

## 验证与判定

- exact GPU/clock/container/commit/binary/command identity 必须闭合；任何 foreign
  process 或 lease 冲突立即停止。
- 正式 measured binary 必须来自 `clean-first` build；exact mode 必须同时闭合 source
  dispatch、64000-loop function 与完整 SASS opcode，benchmark commit 必须可从 canonical
  remote `main` 获取。
- Docker 必须用 `--gpus device=<leased UUID>`，容器内只暴露一个 ordinal 0；执行
  前后同时校验内外 UUID、110 SM、P-state、application/current SM 与 memory clocks。
- `SM120_benchmark` 关键 per-cycle record 的 run-to-run CV `<=1%`；DRAM 各方向
  CV `<=3%`。超过阈值只允许标 `unverified`，不得挑最好一次。
- profile 公式独立重算；per-cycle 值超过相同 instruction 的官方架构值时先审查
  work/count/clock 口径，不截断到 100%。
- exp_024/025 只消费 `vetted` 且 exact/compatible 的 record；跨 GPU 或不同
  instruction mode 只作 diagnostic。
- 更新后 reader-facing ceiling card 以百分比为主，raw rates 转移到 `model.json`。
- 报告动笔前运行 `data-audit`。

## 输出

```text
exp_026_5kp_ceiling_calibration/
  plan.md
  run_remote.sh
  build_result.py
  results/
    result.md                 # 百分比与适用性摘要
    profile.json              # 原始 roof、公式、身份与 evidence locator
    raw/                      # 仅必要原始日志与 telemetry
```

不提交 build tree、CUTLASS checkout、容器日志噪声或重复派生文件。KDK 保存方法，
consumer experiment 保存本次 5KP profile 与 raw evidence。

## Decision

- `accept`：FP4/FP8 per-cycle 与 DRAM profile 均通过身份、变异度和数据审计，
  exp_024/025 可使用兼容记录给出 calibrated ceiling percentage。
- `partial`：部分 record 有效；只更新对应 op，其余保持 unavailable。
- `reject`：GPU/window/benchmark identity 或 work/count 口径不能闭合。

## Plan Review

**Date**: 2026-07-22
**Reviewer**: subagent `/root/exp026_plan_review`

**Verdict**: ✗ Misaligned（以下缺口已一次性修正；按 single-round 规则不复审）

- base benchmark 缺 exp_025 exact E4M3×E4M3 no-scale record：新增并以 SASS 验证，
  capture 使用新 clean commit；
- 单 warp per-cycle 外推不能代表 full-card sustained：增加 NVFP4/FP8 exact-mode
  full-card saturator，区分 instruction、burst 与 sustained-window ceiling；
- DRAM payload 与 target NCU physical traffic scope 不同：增加最小 physical-counter
  校验和读写混合下界公式，atomic/reduction 不宣称完整 memory roof；
- 固定 installed 110 SM，分别定义 useful/executed cycle efficiency，并钉死
  `sm__cycles_elapsed.sum` 同-launch口径；
- 容器按 leased UUID 隔离，并增加内外 GPU/clock/process pre/post gate。
