# RTX 5KP Calibrated Hardware Ceiling

## 结论

本 profile 已通过：**accept**。exp_024 的 NVFP4 与 exp_025 的 FP8 no-scale 现在都有 compatible measured ceiling；报告消费端应优先展示下面的 ceiling 达成率，而不是原始 TFLOP/s / GB/s。这里不建立 operator SOTA。

## Tensor Core ceiling 质量

| 模式 | Per-cycle / architecture nominal（diagnostic） | 8 blocks/SM saturation | ~100 ms roof / app-clock nominal（diagnostic） | 重复稳定性 | measured record 状态 |
|---|---:|---:|---:|---:|---|
| NVFP4 E2M1×E2M1 VS16 | 98.75% | 99.46% | 106.79% | CV 0.033% | ✓ vetted |
| FP8 E4M3×E4M3 no-scale | 99.58% | 99.46% | 107.03% | CV 0.015% | ✓ vetted |
| FP8 E4M3×E4M3 VS32 | 97.12% | 99.58% | 107.01% | CV 0.020% | ⚠ diagnostic |

`~100 ms / nominal` 超过 100% 表示 2377 MHz application-clock nominal 与该测量窗口的有效频率/计数口径尚未闭合；结果与 boost 一致，但现有 telemetry 不能按 kernel 完成归因，因此此列仅作 diagnostic，也不截断为 100%。消费端可用 per-cycle normalization 时优先用 per-cycle record；否则使用同 window 的 full-card record。

`Per-cycle / architecture nominal` 同样是带显式 16-cycle issue 假设的校准诊断；表中的 vetted 指 exact-mode measured record 与证据身份，而不是把 nominal 比值升级为官方 SKU spec。

## DRAM physical ceiling 质量

| 方向 | Physical ceiling / memory-clock nominal | 三次重复稳定性 | 状态 |
|---|---:|---:|---|
| Read | 87.94% | CV 0.035% | ✓ vetted |
| Write | 82.81% | CV 0.100% | ✓ vetted |
| Copy (R+W) | 77.85% | CV 0.034% | ✓ vetted |

DRAM 百分比使用 NCU physical read/write bytes 校准；D2D 仅作 runtime copy diagnostic。Reduction/atomic 不能据此宣称完整 op ceiling。

完整 raw rates、公式输入、telemetry、binary/source digest 与适用范围见 [profile.json](profile.json)。
