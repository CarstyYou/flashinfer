# Experiment 001 Plan: Qwen3.5 fused-MoE backend comparison

## Goal

Measure one FlashInfer CuteDSL fused-MoE latency series and percentage speedup
against CUTLASS FP4 and a vLLM Triton FP8 reference for the Qwen3.5 prefill shape.
Adding Triton completed the same backend-comparison goal and therefore updated
this experiment in place; it did not create a second experiment or result.

## Contract

- Prefill cases: `M={256,512,1024,2048,4096,8192}`.
- Shape: `E=256`, `H=2048`, `I_tp=512`, `topk=8`, SwiGLU, BF16 output.
- Backends:
  - FlashInfer CuteDSL W4A4/NVFP4.
  - FlashInfer CUTLASS fused-MoE NVFP4.
  - vLLM legacy `fused_experts`, Triton tensor-scaled W8A8 FP8.
- Metrics:
  - `Speedup vs CUTLASS = (CUTLASS / CuteDSL - 1) * 100%`.
  - `Speedup vs Triton = (Triton / CuteDSL - 1) * 100%`.
  - Both formulas use the CuteDSL latency from the CUTLASS arm.
- Customer threshold under evaluation: `Speedup vs Triton >= 100%` for every
  prefill case.
- `b12x` is out of scope.

## Measurement sources

CUTLASS and Triton require different native runtimes and were measured in
separate arms. The canonical table contains one CuteDSL series from the CUTLASS
arm and uses it for both speedups. `Speedup vs Triton` is therefore an explicit
cross-arm ratio, not a paired same-host comparison. The CuteDSL repeat from the
Triton arm is retained only as diagnostic evidence and is not an input to the
canonical table or verdict.

### CuteDSL and CUTLASS arm

- CUDA Graph with external events, 192 MiB L2 flush, 5 warmups, 50 iterations,
  5 repeated samples.
- CuteDSL receives BF16 input and quantizes online in the measured path.
- CUTLASS receives input quantized before the timed closure. Its speedup is a
  comparison of the preserved benchmark contracts, not an equal-boundary
  kernel-efficiency claim.
- The historical raw run also captured `M<256`; those rows are provenance only
  and excluded from the canonical result.

### Triton arm

- Same timing parameters: one forward per CUDA Graph replay, external events
  inside the graph, and 192 MiB L2 flush before every replay.
- Triton and the excluded diagnostic CuteDSL repeat consumed the same persisted
  BF16 input, routing tensors, fixture hash, and expert-occupancy hash. The
  canonical CuteDSL series did not use these fixtures.
- Timed region includes online input quantization, route/align work, FC1,
  SwiGLU/intermediate quantization, FC2, and final weighted reduction.
- Top-k selection, weight preparation, JIT/config setup, graph capture, and L2
  flush are outside timing.
- DeepGEMM and CUTLASS block-scaled alternatives are disabled for the Triton
  row. The exact shape has no tuned Triton config and uses the recorded default
  heuristic.

The delivered archive does not identify the customer's historical Triton
quantization recipe. Consequently the FP8 baseline is labeled as the explicit
vLLM legacy tensor-scaled reference and is not presented as a confirmed
production baseline.

## Validation and evidence

1. Require all 18 latency values needed by the canonical table: six cases each
   for the single CuteDSL series, CUTLASS, and Triton.
2. Fail closed on missing/duplicate cases, error rows, timing metadata drift,
   non-finite values, or repeated-sample spread above 5%.
3. For each Triton row, require fixture, occupancy, GPU UUID, output shape
   `[M,2048]`, BF16 dtype, finite values, and nonzero content.
4. Keep `plan.md` and executable scripts at the experiment root. Place every
   script-generated output and report under `results/`; scripts default to
   that location.
5. Preserve raw runs under explicit `results/*_raw.*`, `results/*_initial.*`,
   or `results/*_rerun.*` names. The canonical artifacts are `plan.md`,
   `results/formal.csv`, `results/result.md`, and `results/manifest.md`.
6. Audit every displayed latency and speedup against raw CSV before verdict.

This is performance-only evidence. It makes no numerical-equivalence claim
between FP4 and FP8 and no kernel-level causal claim.

## Decision rule

- **Target met**: all six `Speedup vs Triton` values are at least `100%`.
- **Target not met**: valid evidence exists and any in-scope case is below
  `100%`; report every case rather than averaging.
- **Inconclusive**: wrong/unresolved dispatch, contaminated GPU, source-contract
  mismatch, unstable samples, missing evidence, or unverifiable formula.

## Review history

- Initial CUTLASS arm review: aligned; case identity, timing, isolation, and
  evidence retention were explicit.
- Triton scope audit: initially required revision because the customer FP8
  recipe was unverified, the stock harness was not paired, routing differed,
  and functional sanity was missing. The Triton measurement added explicit
  baseline labeling, persisted fixtures, a common timer, and fail-closed sanity
  checks.
- Reporting follow-up: one CUTLASS-arm CuteDSL series now supplies both
  denominators. The report labels the Triton ratio as cross-arm and retains the
  Triton-arm CuteDSL repeat only as diagnostic evidence.
