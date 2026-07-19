# exp_006 Completion-Anchored Capture Runbook

只在已取得独占 5KP lease、且 exp_004 identity prerequisites 全部满足的 pinned container 中执行。
本 runbook 不申请/修改 KDK lease；`GPU_UUID`、application clocks、container digest、Python dependency
lock 与 source checkout 必须在进入本流程前已固定。所有命令使用 container Python 3.10+，并从本实验目录执行。

## 1. Frontend tests

```bash
python -m pytest -q \
  test_build_completion_probe.py \
  test_capture_completion_timing.py \
  test_analyze_completion_timing.py \
  test_build_evidence.py
```

测试覆盖 339-slot partition、逐 warp `Ai/Ci/Di/Ei/Fi` source anchors、control/probe matched plumbing、
exact-fill/sentinel、每个 same-warp edge 与跨 tile `Fi→A(i+1)` 单调、`max(Ci)≤min(Di)`、声明过的
collective timeline `max(Ai)→max(Ci)→min(Di)→max(Ei)→max(Fi)`、phase ownership audit、
share denominator 与 descriptor contract。

## 2. Build two immutable overlays

两个 output 目录必须不存在；不得复用 exp_004 overlay 或 JIT root。

```bash
python build_completion_probe.py \
  --flashinfer-root "$FLASHINFER_ROOT" \
  --output results/overlays/measurement_no_marker \
  --disabled

python build_completion_probe.py \
  --flashinfer-root "$FLASHINFER_ROOT" \
  --output results/overlays/completion_anchored_probe
```

比较两个 `identity.json`：production hashes、kernel overlay hash、event ABI 与 base-builder hash 必须相同；
只有 dispatch hash/`probe_enabled` 可以因 compile-time marker flag 不同。

## 3. Capture each arm in a fresh process/root

预先设置并核对：

```bash
export CUTE_DSL_KEEP=ir,ptx,cubin,sass
```

然后分别在两个全新 Python process 执行：

```bash
FLASHINFER_WORKSPACE_BASE="$CONTROL_JIT_ROOT" python capture_completion_timing.py \
  --flashinfer-root "$FLASHINFER_ROOT" \
  --arm measurement_no_marker \
  --kernel-overlay results/overlays/measurement_no_marker/moe_dynamic_kernel.py \
  --dispatch-overlay results/overlays/measurement_no_marker/moe_dispatch.py \
  --jit-root "$CONTROL_JIT_ROOT" \
  --output results/raw/measurement_no_marker \
  --expected-gpu-uuid "$GPU_UUID" \
  --warmup 2 --replays 5

FLASHINFER_WORKSPACE_BASE="$PROBE_JIT_ROOT" python capture_completion_timing.py \
  --flashinfer-root "$FLASHINFER_ROOT" \
  --arm completion_anchored_probe \
  --kernel-overlay results/overlays/completion_anchored_probe/moe_dynamic_kernel.py \
  --dispatch-overlay results/overlays/completion_anchored_probe/moe_dispatch.py \
  --jit-root "$PROBE_JIT_ROOT" \
  --output results/raw/completion_anchored_probe \
  --expected-gpu-uuid "$GPU_UUID" \
  --warmup 2 --replays 5
```

每个 `eager_timing.pt` / `timing_*.pt` 必须包含 `task_expert`、`task_m_tile`、
`task_slice_begin`、`task_slice_count`、`task_valid_rows` 与同一 field-major
`descriptor_order_sha256`。任一 source/environment、correctness、workspace、descriptor-order、locked-case、
event exact-fill/sentinel/monotonic/envelope 或 foreign-process gate 失败即停止。

## 4. Post-capture exact static extraction — hard gate

`capture.json.jit_artifacts` 只保证 fresh loadable JIT artifact identity；CuTeDSL 的 exact cubin/PTX/SASS
不保证直接位于 `FLASHINFER_WORKSPACE_BASE`。因此不能用 capture 内 `.cubin/.ptx/.sass` 文件计数代替
static proof。对两个 arm 从各自 fresh provider/cache 输出中提取实际运行的 dynamic MoE code object，保存到：

```text
results/raw/completion_static/measurement_no_marker/
results/raw/completion_static/completion_anchored_probe/
```

至少封存 exact `.cubin`、PTX、`nvdisasm` SASS、`cuobjdump --dump-resource-usage`、提取命令、原始
provider path 与所有 SHA-256。若输入是 loadable `.so`，可用 `cuobjdump --extract-elf all` 提取 code
objects，再以 cache key/kernel symbol 锁定本次 `MoEDynamicKernel`，禁止误选 fp4 quantization helper。

Probe SASS 必须逐个覆盖 compiler unroll/backedge 中所有 marker 实例，并以 timestamp-read 指令（不是延迟的
global store 或 source line）证明：Ai 在 consumer wait 前；Ci 在对应 warp 最后 OMMA 后、首次 accumulator
dependency 前；Di 在 R→S/fence/pre-scatter barrier 后；Ei 在对应 warp scatter loop 后/post barrier 前；
Fi 由同一 warp 在该 barrier 后写入。event gate 必须逐 warp 验证 `Ai≤Ci≤Di≤Ei≤Fi`，以
`min(Di)→max(Ei)` 定义 collective scatter envelope、`max(Fi)` 定义 tile completion，并保存全部
per-warp raw edges；不得恢复代表 warp 或用 timestamp tolerance 掩盖反转。任一 artifact identity、resource extraction 或 boundary proof 失败，
停止 phase 归因并保留证据。

完成 static hard gate 后，才可运行独立 analyzer/data-audit；不得修改或覆盖 exp_004 artifacts。
