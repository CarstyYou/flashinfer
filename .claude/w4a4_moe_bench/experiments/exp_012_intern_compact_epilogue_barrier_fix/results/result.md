# exp_012 结果：跨 pass barrier 不是有效修复

只做 correctness，未运行 benchmark 或 NCU。

| Arm | M | Replay | Formal correctness | Full-zero rows | Workspace gate |
|---|---:|---:|---|---|---|
| 原 Intern fresh control | 8192 | 3 | 0/3 通过 | 1806 / 1799 / 1806 | 3/3 通过 |
| 每个 Q1 pass 后同步 | 4096 | 4 | 0/4 通过 | 1121 / 1124 / 1124 / 1125 | 4/4 通过 |
| 每个 Q1 pass 后同步 | 8192 | 4 | 0/4 通过 | 1808 / 1805 / 1811 / 1806 | 4/4 通过 |
| 每个 Q1 pass 后同步 | 2048 | 4 | 4/4 通过 | 0 / 0 / 0 / 0 | 4/4 通过 |

## 结论

拒绝这个具体修复候选：把原有 `fence_proxy + epilog_sync_barrier` 移到每个 64-row Q1 pass
末尾后，M4096/M8192 仍在全部 replay 中出现大量全零行。因此该单点 barrier relocation
不足以修复 Intern kernel。

本实验没有证明其他位置不存在竞态；具体根因仍未定位，不能继续无证据推测。

## 执行边界

- 每个 M 实际进行一次 fixture/build/capture，随后执行 3 或 4 次 CUDA Graph replay；没有完成计划中的两次独立 preparation。
- `expected_launch` 是 harness contract，不是 profiler-observed geometry；本实验未运行 NCU 或 NSys。
- 候选身份由 exact overlay import、fresh JIT namespace、dynamic cache entry 与唯一 cubin hash 约束。
- 这些证据足以拒绝该 candidate 作为 correctness fix，但不能排除同一区域存在其他竞态。

## 证据身份

- Baseline overlay: `42ca8d40e18b5d0f001236b09b85cbc0aa30e6010f0954efd538d8b9a2fb57d2`
- Candidate overlay: `9aaa5299fc77ee1dbbb3d5cc1426ddef5b46e70704e36609a5059f688379aa01`
- Baseline cubin: `1732f1de50bea28391e07ea12b71512addc8e87b72b5d761cd8de5227ed66df0`
- Candidate cubin: `f7292f5f36c9f29fdc1dfd5dbf81390da2e9ace9cc8254c95bee625ce24acdc4`
- GPU: `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`; nvcc/ptxas 13.2.78
- 原始数据：`raw/baseline_control_m8192.json`、`raw/candidate_m4096_m8192_m2048.json`
