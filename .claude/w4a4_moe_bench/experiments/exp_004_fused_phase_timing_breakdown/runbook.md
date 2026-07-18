# exp_004 Whole-Kernel Runbook

只在独占、通过 identity gate 的 5KP 上执行；固定 pinned container、GPU UUID、application clock、source、
fixture 和 JIT roots。所有命令在本实验目录、container Python 3.10+ 中运行。

## 1. Build immutable overlays

```bash
python build_whole_kernel_probe.py \
  --flashinfer-root "$FLASHINFER_ROOT" \
  --output results/overlays/whole_kernel_probe

python build_whole_kernel_probe.py \
  --flashinfer-root "$FLASHINFER_ROOT" \
  --output results/overlays/whole_kernel_control \
  --disabled
```

Control 与 probe 必须来自同一个 production source；control 只保留相同 plumbing，关闭 marker writes。

## 2. Capture control and probe

```bash
python run_whole_kernel_capture.py \
  --flashinfer-root "$FLASHINFER_ROOT" \
  --arm measurement_no_marker \
  --kernel-overlay results/overlays/whole_kernel_control/moe_dynamic_kernel.py \
  --dispatch-overlay results/overlays/whole_kernel_control/moe_dispatch.py \
  --jit-root "$CONTROL_JIT_ROOT" \
  --output results/raw/whole_kernel_control \
  --expected-gpu-uuid "$GPU_UUID" --warmup 2 --replays 5

python run_whole_kernel_capture.py \
  --flashinfer-root "$FLASHINFER_ROOT" \
  --arm probe_candidate \
  --kernel-overlay results/overlays/whole_kernel_probe/moe_dynamic_kernel.py \
  --dispatch-overlay results/overlays/whole_kernel_probe/moe_dispatch.py \
  --jit-root "$PROBE_JIT_ROOT" \
  --output results/raw/whole_kernel_probe \
  --expected-gpu-uuid "$GPU_UUID" --warmup 2 --replays 5
```

任一 correctness、workspace、event exact-fill/sentinel、GPU identity 或 foreign-process gate 失败即停止。

## 3. Reduce and seal

```bash
python analyze_whole_kernel_timing.py \
  results/raw/whole_kernel_probe/timing_{0,1,2,3,4}.pt \
  --task-tail 2536 \
  --output results/whole_kernel_timing.json

# 从 exact captured cubin 提取 control/probe cubin、PTX、SASS 与 resource_usage，
# 保存到 results/raw/whole_kernel_static/<arm>/ 后再封存。
python finalize_whole_kernel.py
python finalize_whole_kernel.py --check
```

Canonical `manifest.json` 只能由 `finalize_whole_kernel.py` 生成。旧 consumer-only `%clock64`、306-tick、
blocked-finalizer 与 IKET fallback 路径均为 superseded history，不得覆盖当前 whole-kernel closure。
