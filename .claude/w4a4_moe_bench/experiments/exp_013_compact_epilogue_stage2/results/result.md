# exp_013：Stage2 compact epilogue

结论：**Reject，不合入优化 kernel。** v2 已修正正确性，但有 static stack traffic，且相对 exp_008
在两个 case 都稳定变慢。

| M | exp_008 | exp_013 v2 | Speedup |
|---:|---:|---:|---:|
| 256 | 519.099 µs | 522.249 µs | **−0.603%** |
| 8192 | 1567.009 µs | 1579.749 µs | **−0.806%** |

`Speedup = exp_008_time / exp_013_time - 1`。每个 M 为两组 A-B-B-A；每个位置 `warmup=5`、
`timed=50`，每次 replay 前 flush 192 MiB L2。两组在 M256 分别为 −0.625% / −0.582%，在
M8192 分别为 −0.883% / −0.730%，方向一致。

正确性方面，v2 的 canonical M256/M8192、`valid_rows=63/64/65` 与 sparse-empty 共 12 次 replay
全部通过，zero rows 为 0；最坏 relative L2 为 1.63%。初版全零的根因是从 compact `sC[M64]`
错误反推 full FC2 accumulator layout；改用 full tiled-MMA accumulator layout 后恢复正确。

静态资源对比：exp_008 为 `REG=160, STACK=0, LDL/STL=0`；v2 为
`REG=168, STACK=24 B/thread, LDL=5, STL=5`。因此即使忽略小幅性能回退，v2 也不满足 zero-spill
约束。当前证据只判定这个完整 Stage2 compact bundle 无收益，不把回退单独归因于某个 phase。

证据： [correctness_v2.json](raw/correctness_v2.json)、[ABBA summary](perf/summary.json)、
[static resources](static_resources.json)。
