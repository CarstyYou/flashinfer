# exp_011 capture runbook

在 FlashInfer repo 根目录、锁定容器与独占 5KP 租约内执行。

先生成 overlay：

```bash
python .claude/w4a4_moe_bench/experiments/exp_011_route_q0_pack_vs_chain/build_overlays.py \
  --flashinfer-root "$PWD" \
  --output /path/to/exp011_overlays
```

每个 variant 分别使用全新的 JIT root 和 output。`probe` 保存 P0--P4 CTA
时间戳；`no_marker` 保存未插桩完整 fused launch event time：

```bash
python .claude/w4a4_moe_bench/experiments/exp_011_route_q0_pack_vs_chain/capture_arm.py \
  --flashinfer-root "$PWD" \
  --variant identity \
  --mode probe \
  --overlay-root /path/to/exp011_overlays \
  --jit-root /fresh/jit/identity_probe \
  --output .claude/w4a4_moe_bench/experiments/exp_011_route_q0_pack_vs_chain/results/raw/identity_probe \
  --expected-gpu-uuid "$KDK_LEASE_GPU_UUID"
```

`--variant` 依次为 `identity`、`shared_equal_scale`、`static_schedule`、
`precomputed_phys_row`；每个 variant 都运行 `probe` 与 `no_marker`。继承的
exp_004 runtime gate 还要求：

```text
CUDA_VISIBLE_DEVICES=$KDK_LEASE_GPU_UUID
FLASHINFER_WORKSPACE_BASE=<本 arm 的 --jit-root>
CUTE_DSL_KEEP=ir,ptx,cubin,sass
KDK_LEASE_ID=<当前租约>
W4A4_IMAGE_DIGEST=<锁定值>
W4A4_PYTHON_DEPS_SHA256=<锁定值>
```

不得复用 JIT root 或 output；脚本把它们视为 immutable artifact。
