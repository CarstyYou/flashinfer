# exp_014：Scatter 4 Warp → 8 Warp

## 结论

**Accept。** Scatter 已从 W0–W3 各处理 `64×64`，改为 W0–W7 各处理互不重叠的
`32×64` strip。主要 prefill case 的 fused E2E 提升 `5.69%–7.59%`，且未产生 spill。
该实现已提升到 `moe_dynamic_kernel_opt.py`。

## 未插桩 fused E2E

Speedup = `Baseline / Candidate - 1`。

| M | Baseline 4-warp | Candidate 8-warp | Speedup |
|---:|---:|---:|---:|
| 256 | 519.268 µs | 519.391 µs | -0.02% |
| 512 | 536.222 µs | 536.083 µs | +0.03% |
| 1024 | 578.905 µs | 579.697 µs | -0.14% |
| 2048 | 691.697 µs | 642.908 µs | **+7.59%** |
| 4096 | 920.455 µs | 870.934 µs | **+5.69%** |
| 8192 | 1568.567 µs | 1460.144 µs | **+7.43%** |

M8192 的 5 组 A-B-B-A 全部为正收益：`+7.39%–+7.46%`。

## Scatter phase 诊断

| M8192 sampled full-M128 task | Baseline | Candidate | Latency reduction |
|---|---:|---:|---:|
| Scatter body | 1904 ns | 1424 ns | **25.21%** |
| Scatter + post-sync | 1936 ns | 1440 ns | **25.62%** |

这是 matched `%globaltimer` 诊断插桩，只覆盖 task slot 0 的 16 个 output tile；最终性能判定以上面的未插桩 E2E 为准。

## 正确性与资源

| Gate | 结果 |
|---|---|
| GPU correctness | 两臂各 12 个 case 全部通过 |
| Scatter ownership | 48 个 case 通过；full tile 的 W0–W7 均有工作且无重漏 |
| Registers/thread | 160 → 165 |
| Static spill | 两臂均为 0 |
| Dynamic spill | M8192 两臂均为 0 |

完整身份、原始数据哈希与限制见 [evidence.json](evidence.json)、[manifest.json](manifest.json) 和
[scatter_phase_evidence.json](scatter_phase_evidence.json)。Baseline dynamic NCU 复用了 exp_015 的
exact-cubin capture，并重新校验了 source、cubin、GPU、launch 与 tensor-work identity。
