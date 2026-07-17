# exp_004 Runbook

只允许在独占、通过 identity gate 的 5KP 上执行：GPU name 必须为 `NVIDIA Graphics Device`、110 SM、SM12.0/12.1；`RTX PRO 6000*` 不得替代。不要占用 exp_005 的 GPU。

## 前置身份

在 pinned container 内设置并核对：

```bash
export KDK_LEASE_ID=<exclusive-lease-id>
export W4A4_IMAGE_DIGEST=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
export W4A4_PYTHON_DEPS_SHA256=32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74
export GPU_UUID=<leased-5kp-gpu-uuid>
export KDK_LEASE_GPU_UUID="$GPU_UUID"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
```

`results/overlays/identity.json` 已由 production kernel/dispatch SHA 的 exact transforms 生成。若 source 漂移，停止；不要复用旧 overlay。

## 执行顺序

以下命令均在本实验目录执行，使用 container 的 Python 3.10+：

```bash
python build_overlays.py --flashinfer-root "$PWD/../../../../" \
  --output-dir results/overlays

python run_exp004.py --flashinfer-root "$PWD/../../../../" prepare \
  --expected-gpu-uuid "$GPU_UUID"

python run_exp004.py correctness

python run_exp004.py --flashinfer-root "$PWD/../../../../" capture-phases \
  --expected-gpu-uuid "$GPU_UUID" --warmup 5

python run_exp004.py --flashinfer-root "$PWD/../../../../" calibrate \
  --expected-gpu-uuid "$GPU_UUID" --warmup 3 --samples 4096

for arm in normal_no_marker measurement_no_marker probe_candidate; do
  python capture_ncu.py --flashinfer-root "$PWD/../../../../" \
    --expected-gpu-uuid "$GPU_UUID" --arm "$arm" --warmup 5
done

python build_ncu_evidence.py
python build_binary_identity.py
python analyze_phase_timing.py
python build_result.py
python run_exp004.py refresh-manifest
```

每一步都是 fail-closed。`build_binary_identity.py` 非零退出时仍会保留 binary gate；若 stack/spill/resource 或 non-probe semantic projection 漂移，不得继续解释 phase 百分比。`analyze_phase_timing.py` 返回 `3` 表示只允许 instrumented diagnostic share，不能写成 production-representative share。

若 probe 在 preparation 阶段已触发 resource/spill、correctness 或 event-contract immediate-stop，不再执行后续 capture。使用 fresh replay 自动落盘的失败记录收口：

```bash
python build_blocked_preflight.py \
  --probe-failure "$PROBE_FAILURE/preparation_failure.json"
python finalize_blocked.py
```

该路径只生成 compact blocked evidence、`result.md` 与 `manifest.json`；不得生成或补造 phase share。

## IKET fallback

本机没有已审计的 `run-iket`/IKET provider，当前 fallback 不可执行。只有 primary clock probe 因可解释的 plumbing/capture feasibility 失败时，才允许按 `plan.md` 第 8 节启用 KDK `iket_safe_capture.py`；必须另建 IKET-compiler no-marker control 与 coarse-marker candidate，并重新通过同一三身份 resource/spill/SASS gate。旧 cadence overlay 的 `STACK 488→432 B/thread` 结果禁止复用。
