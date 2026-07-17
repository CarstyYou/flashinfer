# exp_004：Fused Phase Timing Breakdown

## 结论

实验按预注册门槛停止：clock probe 改变了 production 的 stack/spill 身份；本次 probe replay 的 reference correctness 未通过，运行时事件槽位也没有写入。结论为 `measurement perturbation prevented formal timing`；不发布任何 phase 占比。

## 失败门槛

| Gate | 结果 | 证据 |
|---|---:|---|
| Production / measurement control 静态身份 | PASS | REG、STACK、SMEM、local SASS 与 semantic projection 一致 |
| Probe resource / spill 身份 | FAIL | 见下表；probe 不再代表 production spill 结构 |
| Probe reference correctness | FAIL | cosine `0.997062`；relative-L2 `0.076544`；max-abs `0.869024` |
| Probe event contract | FAIL | ticks `0/776016`；CTA map `0/2536` |
| IKET fallback | UNAVAILABLE | provider `run-iket`；要求版本 `0.7.10`；observed `None` |

## 静态证据

| Arm | REG/thread | STACK B/thread | Static SMEM B/CTA | STL | LDL | Spill annotations |
|---|---:|---:|---:|---:|---:|---:|
| `normal_no_marker` | 255 | 488 | 1024 | 68 | 122 | 190 |
| `measurement_no_marker` | 255 | 488 | 1024 | 68 | 122 | 190 |
| `probe_candidate` | 255 | 456 | 1024 | 64 | 132 | 196 |

## 解释边界

- Probe PTX 文本中检出 `28` 个 clock64 occurrence 与 `1` 条静态 probe-store 指令；这只证明插桩已 lowering。
- 运行时 0-write 的具体原因尚未定位；不归因于 cache、pointer、CUDA Graph 或其他机制。
- Immediate-stop 后不继续 cross-arm correctness、phase capture、calibration 或 NCU；没有合法的 phase timing 数据可解释。
