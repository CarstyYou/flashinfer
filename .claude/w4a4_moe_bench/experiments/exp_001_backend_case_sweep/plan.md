# Experiment 001 Plan: Qwen3.5 prefill MoE backend case sweep

## Goal

Measure one fresh FlashInfer CuteDSL W4A4 latency series against:

1. FlashInfer CUTLASS NVFP4 with a matched BF16-input/online-quantization
   operator boundary; and
2. the SGLang Triton tensor-scaled W8A8 FP8 implementation.

The customer threshold is
`(SGLang Triton FP8 / CuteDSL FP4 - 1) * 100% >= 100%` for every case.
This follow-up replaces the invalid prequantized-CUTLASS/vLLM evidence in the
same experiment; it does not create another experiment.

## Fixed scope

- Cases: `M={256,512,1024,2048,4096,8192}` only (prefill).
- Shape: `E=256`, `H=2048`, `I_tp=512`, `topk=8`, SwiGLU, BF16 output.
- `b12x` and `M<256` are out of scope.
- One persisted deterministic BF16 input/routing fixture per M is shared by
  all three backends. The expert-weight shape, seed, and value distribution
  are fixed; each precision uses its native offline weight quantization.
- Each fixture stores contiguous `x[M,2048]`, `topk_ids[M,8]` int32 and
  `topk_weights[M,8]` FP32. IDs are unique per token and sorted by descending
  source logit; weights are finite, nonnegative, and normalized to one per
  token. The NPZ file plus each logical tensor and expert-occupancy vector are
  SHA-256 hashed and revalidated in both runtimes.
- Weight preparation, routing/top-k generation, JIT/config selection, graph
  capture, and L2 flush are outside timing. Online activation quantization is
  inside each measured operator boundary.

## Comparison registry

### A. CuteDSL FP4 vs CUTLASS FP4 (paired)

- Target: `flashinfer.fused_moe.cute_dsl.B12xMoEWrapper`, BF16 input -> fused
  W4A4 MoE -> BF16 output.
- Baseline: `flashinfer.fused_moe.cutlass_fused_moe`, BF16 input with
  `input_sf=None` -> native online input quantization, FC1, SwiGLU/requant,
  FC2/finalize -> BF16 output.
- Both arms run in one process, on the same leased GPU, from the same fixture
  and canonical weights, with identical timing parameters. Repeat order is
  alternated. This relationship may support a paired performance claim.

### B. CuteDSL FP4 vs SGLang Triton FP8 (cross-runtime)

- Target latency is exactly the CuteDSL value from comparison A; no second
  CuteDSL timing series may enter the canonical result.
- Baseline is SGLang
  `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`
  with BF16 input, FP8 E4M3 weights, per-expert tensor scales,
  `use_fp8_w8a8=True`, dynamic activation scales, and `block_shape=None`.
- The exact FP8 contract is `torch.float8_e4m3fn` contiguous row-major weights:
  `w1[E,2I,H]`, `w2[E,H,I]`; FP32 per-expert tensor scales `w1_scale[E]` and
  `w2_scale[E]`; `per_channel_quant=False`; `a1_scale=None` and
  `a2_scale=None`. The two `None` values mean separate dynamic tensor-wise
  activation quantization for BF16 input and post-SwiGLU BF16 intermediate,
  both inside the measured chain. Harness source hashes lock weight generation,
  quantization, scale orientation, and dequantization.
- The direct callable fixes dispatch to SGLang's legacy Triton fused-MoE
  sequence and bypasses runner selection, DeepGEMM, and CUTLASS. A profiler
  smoke must also record the observed Triton MoE kernel names; unresolved or
  alternate dispatch invalidates this relationship.
- This is a same-GPU, same-fixture, same-measurement-protocol, different-runtime
  comparison. It supports the stated end-to-end operator speed ratio, not a
  fusion-only or precision-only causal claim.

## Locked environments and identity

- FlashInfer production source: commit `074d93e4aa54c75bee1b3dfdb39b7f075a3ff2af`
  (the later experiment-only commits do not alter production kernels).
- CUTLASS submodule: `b46b16d003484063bca4ed365e44095c4c6ed633`.
- FlashInfer runtime: `nvcr.io/nvidia/pytorch:26.05-py3`, image digest
  `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`,
  CUTLASS DSL 4.6.0 dependency overlay digest
  `32090c58dc24660cb2cb6a4368c7fe756a181234055a9bae7f798ebf25d64c74`.
- SGLang runtime: `lmsysorg/sglang:latest` pinned by image digest
  `sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2`;
  packaged SGLang commit
  `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`, PyTorch 2.11.0+cu130,
  Triton 3.6.0.
- Both arms must record full container/source/dependency hashes, the same full
  GPU UUID and KDK lease ID, fixture hashes, protocol lock, artifact/JIT lock,
  and one common fresh rerun ID. Any drift or mixed rerun ID fails closed.
- Each FlashInfer run uses a dedicated initially empty JIT workspace with
  `FLASHINFER_CUTEDSL_IKET_OVERLAY=0` and `FLASHINFER_NVFP4_4OVER6=0`.

## Measurement protocol

- One forward per CUDA Graph replay with external CUDA events recorded inside
  the graph.
- 192 MiB L2 flush before each replay and outside the measured interval.
- Every sample follows `flush kernel -> device synchronize -> graph replay ->
  device synchronize -> elapsed_time(start,end)`. Thus neither an unfinished
  flush nor queued work can enter the graph event interval.
- 5 warmups, 50 measured iterations per repeat, 5 repeats; report median.
- FlashInfer arms alternate order on every repeat. SGLang uses the identical
  warmup/iteration/repeat/flush protocol on the same leased GPU.
- Fail if a case is missing/duplicated, a sample is non-finite, repeat spread
  exceeds 5%, GPU identity changes, a foreign workload is detected, or an
  environment/artifact/protocol identity differs.

## Correctness and dispatch gates

Before timing each case:

1. require output `[M,2048]`, BF16, finite, and nonzero;
2. qualify both FP4 arms against the existing dequantized NVFP4 PyTorch oracle;
3. qualify SGLang against a PyTorch oracle built from the *actual* FP8 weight
   tensors/scales and simulated dynamic E4M3 quantize/dequantize round trips
   for both activations; and
4. capture dispatch evidence for all three arms. Per M, save the resolved
   callable/config plus observed kernel sequence. CUTLASS must show its online
   quant/expand and MoE chain; CuteDSL must show its fused kernel; SGLang FC1
   and FC2 names must match its Triton MoE allowlist, with route/activation/
   reduce helpers allowed. Any backend/fallback name outside the recorded
   allowlist invalidates that M.

The FP4 formal gate is the existing criterion: at least 97% of elements satisfy
`abs_err < max(0.05, 1.5 * oracle.std)` or `rel_err < 0.5`. The FP8 gate uses
SGLang's BF16 test tolerance `rtol=0.1, atol=0.01`; it also records cosine,
relative-L2, maximum absolute/relative error, and percent within tolerance.

FP4 and FP8 outputs are not asserted numerically equivalent to each other.
Failure of any per-arm oracle or dispatch gate stops publication.

## Evidence and reporting

- Executable scripts remain at the experiment root; every generated artifact
  is written under `results/`.
- Move the old vLLM/prequant result set under
  `results/superseded_vllm_prequant/`. It is provenance only and must not be
  read by the canonical builder.
- Canonical inputs are fresh raw/summary/correctness/identity evidence for the
  two relationships. `formal.csv`, `result.md`, and `manifest.md` are rebuilt
  only after all gates pass.
- Fresh evidence is isolated under `results/pair/` and
  `results/sglang_triton/`, both carrying the same rerun ID. The builder rejects
  mixed IDs, unexpected input paths, missing/duplicate M, incomplete paired
  repeats, or any old-format row. Comparison A retains repeat-level pairing;
  comparison B is explicitly a non-paired cross-runtime ratio.
- Report both formulas per M:
  - `(CUTLASS BF16 chain / CuteDSL - 1) * 100%`
  - `(SGLang Triton FP8 / CuteDSL - 1) * 100%`
- Audit every displayed latency and derived speedup back to the fresh raw CSV.

## Decision rule

- **Target met:** all six SGLang speedups are at least 100%.
- **Target not met:** valid evidence exists and any case is below 100%; report
  every case rather than an average.
- **Inconclusive:** any correctness, dispatch, stability, contamination,
  identity, fixture, or evidence-lineage gate is unresolved.

## Plan Review

- Verdict: `NEEDS_REVISION` (single formal pre-execution review).
- Critical gaps found: the initial revision did not fully lock E4M3 tensor and
  scale semantics, oracle threshold, flush/replay synchronization, three-arm
  dispatch proof, fixture layout, or mixed-evidence rejection.
- Required changes applied above: exact tensor/scale/activation contract and
  harness hashes; actual-quantized-tensor oracle with explicit thresholds;
  synchronized replay order; per-M callable/config/kernel evidence for all
  arms; fully hashed normalized fixtures; run-ID-isolated evidence with strict
  six-case/paired-repeat validation.
