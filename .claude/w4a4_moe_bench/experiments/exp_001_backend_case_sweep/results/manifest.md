# Experiment 001 Evidence Manifest

## Canonical scope

- Goal: compare CuteDSL W4A4 with CUTLASS FP4 and a vLLM Triton FP8
  reference for Qwen3.5 fused-MoE prefill.
- Decision cases: `M={256,512,1024,2048,4096,8192}` only.
- The Triton measurement was a same-goal extension and updated this experiment
  in place. [`../plan.md`](../plan.md), [`result.md`](result.md),
  [`formal.csv`](formal.csv), and this file are the only current canonical
  artifacts; there is no nested follow-up result.
- `plan.md` and executable scripts remain at the experiment root. Every
  script-generated output and report lives under this `results/` directory.
- Raw measurements are immutable. Replaced or exploratory runs remain under
  explicit names and are marked below as `superseded`, `diagnostic`, or
  `not-for-verdict`.
- The initial CUTLASS-arm command also measured `M<256`. Those rows remain in
  the raw CSV for provenance but are `not-for-verdict`.

## Canonical artifacts

| Artifact | SHA-256 | Status |
|---|---|---|
| [`../plan.md`](../plan.md) | `c503d394b96023892e9f7602276226accf35af94fc5784eaf958837c1110429a` | canonical contract |
| [`formal.csv`](formal.csv) | `cfdaedd4111c15383f8334fc033798e0a706f03839e7a3700113b9a0d11d3b4c` | canonical data |
| [`result.md`](result.md) | `5f11b94fdc781085a52d7eb6278d8b244ba21426618898207d5707bb58dcd60e` | canonical verdict |

This manifest is the canonical lineage record and therefore does not contain a
self-referential hash.

## Reproduction tooling

| Artifact | SHA-256 | Role |
|---|---|---|
| [`../build_result.py`](../build_result.py) | `f77878860f538b30353868b4065dd58be77e841c67f5f567e95d194b45cfccd0` | canonical fail-closed join |
| [`../test_build_result.py`](../test_build_result.py) | `3ca7551f8d0db0ffdc756ac72f391fe52540b76a38ed6aabea99dc81abdc627c` | join/formula/layout tests |

`formal.csv` is generated from the two active raw CSVs below. The join requires
three measurement sets: the single CuteDSL series and CUTLASS from the CUTLASS
arm, plus Triton from its own arm. All require matching contract fields, finite
samples, and spread at or below 5%.

## Raw evidence lineage

### Active, vetted inputs

| Artifact | SHA-256 | Role |
|---|---|---|
| `cutlass_arm_raw.csv` | `00fc3425ea46f8e476db75352b0ef35f0e844da982cd07a8f1a6264dbaadf0bb` | canonical CuteDSL and CUTLASS; only M>=256 enters verdict |
| `triton_arm_raw.csv` | `6f28a724f6431a59dda1badc03f422394b4bebb6565baa5ee31a460e04c10181` | Triton arm with vetted M=8192 rerun merged in |
| `triton_arm_initial.csv` | `0af6639fef0766754dc4257500c99e3c40a8ee81d7c0e2c427f167c883da64cf` | active source for canonical M=256..4096; only its M=8192 row is superseded |
| `triton_arm_m8192_rerun.csv` | `0003aa9ab88575bdf1f7bacff27f7d2e9fe395385477fa8fc29724e12a94d5a7` | active source for canonical M=8192 Triton row |
| `fixture_manifest.json` | `7feae803bdddbed990aa869e9357422c57bded0e35b86c13db25519b453c50b2` | per-M persisted identity for Triton rows and diagnostic CuteDSL repeat |

Supporting raw logs:

- `cutlass_arm_raw.log`:
  `febd1a2a14a1eee49084eb3e8b07d65653b982e31de3e88b4ea7e565ece08cc4`
- `triton_arm_initial.log`:
  `3b48d9f9798d4a14d07fb94d36404124d2884c3d9ce2becc4cefff87b0f7eb77`
- `triton_arm_m8192_rerun.log`:
  `e45846d869694abbb6210817c915138738155b03c3b81fce6c1c410302c82223`

### Retained but excluded from the verdict

| Artifact | SHA-256 | Status and reason |
|---|---|---|
| `triton_arm_cutedsl_raw.csv` | `ccd87934cdb550f3dfd239484d02cc42a65c769ee932f48afea64f717164875d` | diagnostic bridge repeat; not used in `formal.csv`, speedup, or verdict |
| `triton_arm_cutedsl_initial.csv` | `76150095d1402c1c14d4bc0fce4a2bcfefb277ce85bb2a1208739829360ab8d8` | diagnostic independent repeat |
| `cutlass_arm_exploratory.csv` | `b432ddc720bd34dcb20ca2b53e46e0c18345e53107fcebb12c59bd827b3adca6` | diagnostic |
| `cutlass_arm_smoke_cutedsl.csv` | `c3eccdd7bf1e01888dc77f53ef9bc6eaa2e756dfa9c06036154b1a5c2026106a` | diagnostic smoke |
| `cutlass_arm_smoke_cutlass.csv` | `fb6cf70a57b89717dc75cee94be1f2cf407fcd3817500128b58aeaa3dddecc2e` | diagnostic smoke |

Diagnostic CuteDSL log: `triton_arm_cutedsl_raw.log`,
`f5c23ca2432d2c3b37b1f0ebf565b942924e5f166a728a34b6cd557157aec870`.
Diagnostic CUTLASS exploratory log: `cutlass_arm_exploratory.log`,
`341c50e12b4053cc69867ed9e09d179896d79ea32fdbe224be80ba5607ec950a`.

## Shared contract

- Shape: `E=256`, `H=2048`, `I_tp=512`, `topk=8`, SwiGLU, BF16 output.
- Timing: one fused-MoE forward per CUDA Graph replay; CUDA events inside the
  graph; 192 MiB L2 flush before each replay and outside the timed interval;
  5 warmups, 50 iterations, and 5 repeated samples.
- Speedup: `(baseline time / CuteDSL time - 1) * 100%`; both baselines use the
  one CuteDSL series from the CUTLASS arm.
- Triton was measured on a different host and runtime. Its speedup is therefore
  a declared cross-arm ratio, not a paired same-host comparison. The excluded
  Triton-arm CuteDSL repeat differs from the canonical series by at most 1.24%.
- Performance-only evidence: no FP4/FP8 numerical-equivalence claim.

## CUTLASS arm identity

- Date/host: 2026-07-15, `R6KD-CX8aaS-GPU-16` (`10.6.142.16`).
- GPU: SM120 NVIDIA Graphics Device,
  `GPU-2fdb0b79-0ba7-f356-b714-6c461b71ce12`, driver `580.95.05`.
- Container: `nvcr.io/nvidia/pytorch:26.05-py3`, image
  `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`.
- Runtime: PyTorch `2.12.0a0+5aff3928d8.nv26.05`, CUDA `13.2`,
  NVIDIA CUTLASS DSL `4.6.0`, apache-tvm-ffi `0.1.11`.
- CUTLASS input quantization occurs before its timed closure; CuteDSL online
  input quantization is inside its timed closure.
- No foreign compute process was present on the selected UUID at launch; the
  direct-SSH pool did not provide a hard scheduler lease.

Source identity:

- FlashInfer base: `517cca9c2e7d91f524fcb5f078370c056308d461`.
- Archived benchmark script:
  `40add7573a13888c5aa0eda64fe1b9b8b381338a4b671bdbb7e6655cb582cc60`.
- CuteDSL wrapper:
  `bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4`.
- Dynamic kernel:
  `94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106`.
- Dispatch:
  `cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4`.
- CUTLASS Python/C++ orchestration:
  `ba2478746d1c5de82f2aa05ad6af1bed64be25baac87804ae06a02485c30fe42` /
  `fd24f5f8234b0736f205dd2540f47dcaf90783a53c2fbbab66d0490c9494dbac`.

## Triton arm identity

- Date/host: 2026-07-15, `R6KD-CX8aaS-GPU-09` (`10.6.142.9`).
- GPU: SM120 NVIDIA Graphics Device,
  `GPU-1c189c11-e797-a795-cefd-495b190afebc`, driver `580.95.05`.
- Triton and the excluded diagnostic CuteDSL repeat consumed identical per-M
  BF16 input, normalized routing, fixture SHA-256, occupancy SHA-256, and GPU
  UUID. The canonical CuteDSL series came from the CUTLASS arm on a different
  host and did not use these fixtures.
- CuteDSL image: `cutedsl_460_public:local`, image
  `sha256:b9b9c43432f7513aac416685f4c5b3b58be04bba40bfc9743858b893304a7ecd`;
  PyTorch `2.11.0a0+eb65b36914.nv26.02`, CUDA `13.1`, FlashInfer
  `0.6.12`, CUTLASS DSL `4.6.0`; IKET overlay disabled.
- Triton image: `yanwa/hyimage3-lowbit-vllm:base-20260609`, image
  `sha256:a65a6391c3e548574368a215cd4794e489a56fb699df2990f308f18bf8685623`;
  vLLM `0.11.1rc1`, PyTorch `2.8.0+cu128`, CUDA `12.8`, Triton `3.4.0`.
- Triton recipe: legacy tensor-scaled W8A8 with per-expert E4M3 weight scales
  and BF16 activation quantized inside `fused_experts`. DeepGEMM and CUTLASS
  block-scaled alternatives were disabled.
- No tuned config existed for this exact shape. The recorded default heuristic
  used `BM16/BN32/BK64/G1` at M=256 and `BM64/BN64/BK32/G8` at M>=512.
- Installed vLLM identities: `fused_moe.py`
  `cf23444501ab25500aebd572813251b843ea27eebe286bb1e493aee7631415dc`,
  `config.py`
  `8dce29cafeb88a603daa00bf5c4f8a736ddc79e78d26db8ad704cdbb5cfa4a65`,
  and `vllm/_C.abi3.so`
  `591f2b1c2994765f28b577198c94070050bafbd84af7479ac2601309727467a7`.
- No foreign compute process was present immediately before or after the
  formal runs; the direct-SSH host did not provide a hard lease.

The delivered archive does not prove that this vLLM recipe is the customer's
historical production baseline, so the result labels it only as the explicit
legacy tensor-scaled reference.

## Stability and archive identity

- The initial Triton M=8192 spread was 5.05%, so only that row was rerun. The
  replacement spread is 4.87%, and its median differs by 0.49%.
- All other canonical Triton spreads are below 0.84%; canonical CuteDSL spreads
  are below 0.12%, and canonical CUTLASS spreads are below 1.15%.
- The excluded Triton-arm CuteDSL series differs from canonical CuteDSL by at
  most 1.24%; its two full diagnostic repeats differ by at most 0.20%.
- Source archive: `w4a4_moe_bench_20260714.tar.gz`, SHA-256
  `e94ab39b28b1664e23bb49b6e0d72e0da349cd0e5126d13365c3e76db4ce66f2`.
- The archive's withdrawn historical W8A8 table is diagnostic only and is not
  an input to `formal.csv` or the verdict.

## Adapter identities

- [`../bench_cutedsl_fp4.py`](../bench_cutedsl_fp4.py):
  `793419a0e3928be37b629ac19786a794c0050f74b230166eceda44bb0c23b054`.
- [`../bench_triton_fp8.py`](../bench_triton_fp8.py):
  `5aa5de48aaefd7ab2ce23fc80db86601696122745d5e174107a83816fc9fc7c4`.
- [`../fixture.py`](../fixture.py):
  `b886ce1c38fd02e1f79945a7b06de87b1464f7b93f3b209b5474818e901a4cfb`.
- [`../make_fixtures.py`](../make_fixtures.py):
  `16a9b0ac9df1ed47a4fb8165029c9afad4e452824e74d5bec4af3d18a8582b89`.
- [`../merge_triton_rerun.py`](../merge_triton_rerun.py):
  `8f865785be22464be9310f302dc9cb804f0944efe0027fe6240deb4dccbc9856`.
