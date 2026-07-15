# Experiment 001 Result: CUTLASS vs CuteDSL vs Triton

**Verdict:** the evaluated `100%` CuteDSL speedup threshold versus the vLLM
Triton FP8 reference is **not met** for all prefill cases. `M=256`, `M=8192`
fail; `M=512`, `M=1024`, `M=2048`, `M=4096` pass.

`Speedup = (baseline time / CuteDSL time - 1) * 100%`.

| M | CuteDSL (us) | CUTLASS (us) | Speedup vs CUTLASS | vLLM Triton FP8 (us) | Speedup vs Triton | 100% threshold |
|---:|---:|---:|---:|---:|---:|:---:|
| 256 | 529.357 | 552.369 | 4.35% | 847.223 | 60.05% | FAIL |
| 512 | 551.044 | 570.518 | 3.53% | 1650.833 | 199.58% | PASS |
| 1024 | 604.466 | 603.039 | -0.24% | 1750.087 | 189.53% | PASS |
| 2048 | 732.929 | 721.847 | -1.51% | 1888.346 | 157.64% | PASS |
| 4096 | 1001.441 | 1006.863 | 0.54% | 2154.085 | 115.10% | PASS |
| 8192 | 1738.950 | 1678.002 | -3.50% | 3297.283 | 89.61% | FAIL |

Both speedups use the single CuteDSL measurement from the CUTLASS arm.
Positive speedup means the baseline is slower than CuteDSL.

## Scope

- Prefill only: `M={256,512,1024,2048,4096,8192}`.
- CUTLASS and Triton were measured in separate arms on the same GPU class.
  Speedup vs Triton is therefore a declared cross-arm ratio, not a paired
  same-host comparison.
- Canonical CuteDSL and Triton do not share recorded fixture/routing
  identity; the CUTLASS-arm CSV does not carry those identity hashes.
- CUTLASS excludes BF16 input quantization from its timed closure; CuteDSL
  includes online input quantization.
- Triton is vLLM `0.11.1rc1` legacy tensor-scaled W8A8 using an untuned
  default heuristic, not a confirmed customer production recipe.
- Measured Triton deltas versus the withdrawn historical W8A8 table
  (not used in speedup calculations):
  - `M=256` +15.55%, `M=512` +123.30%, `M=1024` +129.34%
  - `M=2048` +122.29%, `M=4096` +95.38%, `M=8192` +91.26%
- This is performance-only evidence and makes no FP4/FP8 numerical-equivalence
  claim.

## Evidence

- [`formal.csv`](formal.csv): canonical three-backend result data.
- [`cutlass_arm_raw.csv`](cutlass_arm_raw.csv): CUTLASS/CuteDSL raw arm.
- [`triton_arm_raw.csv`](triton_arm_raw.csv): Triton FP8 raw arm.
- [`manifest.md`](manifest.md): setup, source, stability, and evidence identity.
- [`plan.md`](../plan.md): current experiment contract.
