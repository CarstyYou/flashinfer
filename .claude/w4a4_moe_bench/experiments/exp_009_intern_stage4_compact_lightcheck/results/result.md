# exp_009：Production / Intern / exp_008 benchmark

| M | Production | Intern Stage4 compact | exp_008 | Intern vs Production | Intern vs exp_008 | exp_008 vs Production |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 541.384 µs | **490.041 µs** | 519.224 µs | **+10.48%** | **+5.96%** | +4.27% |
| 512 | 567.908 µs | **491.076 µs** | 535.995 µs | **+15.65%** | **+9.15%** | +5.95% |
| 1024 | 620.211 µs | **504.612 µs** | 578.551 µs | **+22.91%** | **+14.65%** | +7.20% |
| 2048 | 755.741 µs | **583.245 µs** | 691.217 µs | **+29.58%** | **+18.51%** | +9.33% |
| 4096 | 1043.561 µs | **Invalid，未测性能** | **921.592 µs** | — | — | +13.23% |
| 8192 | 1805.128 µs | **Invalid，未测性能** | **1568.147 µs** | — | — | +15.11% |

`Speedup = Production / Candidate - 1`。

Intern 在 `M256–2048` 正确性通过；从 `M4096` 开始失败。Invalid 不是普通精度波动：3 次复跑中，
M4096 每次有 1124–1125 行全零，M8192 每次有 1803–1806 行全零；输出有限且 workspace gate 通过。

每个单元格保留 1 个 benchmark sample；sample 内部为 `warmup=5`、`timed=50` 的 CUDA Graph E2E replay，
每次 replay 前执行 192 MiB L2 flush。同一块 5KP、application clock 2377 MHz；benchmark 未使用 NCU 计时。

既有轻量 spill 检查已收口：exact cubin 的 static `STACK/LOCAL/LDL/STL/SpillRefill` 均为 0；
correctness-gated M256 graph node 的 dynamic spill/refill counters 也均为 0。本次 M sweep 未新增 NCU。

机器可校验汇总与原始 JSON 见 [full_m_sweep/summary.json](full_m_sweep/summary.json)；spill 证据见
[static_spill_evidence.json](static_spill_evidence.json) 与 [dynamic evidence](evidence/dynamic_spill/evidence.json)。
