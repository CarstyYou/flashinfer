# exp_020：4-CTA DSM 8-way Scatter Demo

## 结论

**Reject，不集成到 Opt。** DSM-8 correctness 通过，但 device-only kernel wall 从约
`870.4 µs` 增至约 `9994.0 µs`，约为 Direct-32 的 `11.48×`；按
`Direct / DSM - 1` 计算为 **-91.29%**。

## 性能

固定 `M=8192, E=256, H=2048, topk=8`，同一 SM120 5KP、2377 MHz application clock、
同一进程；每臂 warmup 20 次，每组各 100 个 CUDA-event samples。循环内调用预编译的
CuTeDSL launcher，event 只覆盖 device kernel，output clear 与 L2 eviction 在 event 外。

| 顺序 | Direct-32 median | DSM-8 median | `Direct / DSM - 1` | Direct / DSM CV |
|---|---:|---:|---:|---:|
| AB | 870.400 µs | 9992.192 µs | -91.289% | 0.690% / 0.354% |
| BA | 870.400 µs | 9995.776 µs | -91.292% | 0.126% / 0.339% |

两种首发顺序方向一致且各臂 CV 均低于 1.5%。接受门禁要求五组全部正收益；AB、BA
均稳定为负后，剩余三组不再执行，因为已经不可能改变 Reject 判定。

## 正确性与实验身份

- Canonical M8192 与 `valid_rows=1/31/32/33/63/64/65/95/96/97/127/128` tail cases
  的 Direct-32、DSM-8 均通过预设 tolerance。
- Benchmark 复用了 source/environment identity 一致的 validation：kernel source、fixture、group identity、
  container image、GPU UUID 等八项已有检查全部通过。
- Launch 固定为 grid `2548×1×1`、block `288×1×1`、cluster `4×1×1`，每 CTA dynamic
  SMEM `84,992 B`；采样前后只有本实验一个 compute process。
- 静态工作量账本中，设计将 BF16x8 REDG 从 `67,108,864` 降至 `16,777,216`，同时引入
  约 1 GiB partial reads（其中约 768 MiB 为 remote DSM）、约 4.03 亿次 FP32 add 和逐 tile
  cluster synchronization。

## 边界

本实验只证明这个完整 DSM-8 机制组合明显回退，足以拒绝集成。它没有进一步区分 remote DSM
load、FP32 merge、cluster barrier 或 cluster residency 各自贡献多少；由于性能门禁已失败，本轮按
计划不启动 NCU/phase breakdown，也不据此编造单一根因。

Correctness 与 benchmark 是两个独立进程；旧 validation manifest 已被后续结果覆盖，因此当前证据没有
显式绑定两次运行的 cubin/PTX、runner 与依赖哈希，只能确认上述 source/environment identity。该缺口会
阻止 Accept，但不会推翻两个顺序下约 `11.48×` 回退所要求的 Reject。

最初曾用 `@cute.jit` Python wrapper 直接置于 CUDA events 之间，错误地把 host dispatch 空档计入
时间；该轮数据已排除。权威数据仅来自本报告对应的 compiled-launcher raw samples。

证据：[raw/demo.json](raw/demo.json)；[manifest.json](manifest.json)。
