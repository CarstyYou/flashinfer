# Draft: reply to flashinfer issue #3765 (moe_gemm naming / backend folding)

> 状态: 已由 xiy 发到 issue (2026-07-09)。留档。
> 结构 (per xiy): 不做合并与否的决定, 不提 naming simplify;
> 说明实际差异只在 SFA/SFB layout (对齐 DG style, quant 函数也对齐 DG), 征求对方意见。

---

Author of #3562 here — some context that may help this discussion.

The A / B tensor layouts are actually the **same** between the two entries: token-packed
A `(cum_m, k)` with an `m_indptr` group descriptor (no token padding), and B
`(num_experts, n, k)`. The real difference is the **scale-factor layouts** and the
quantization helpers that produce them:

| | `grouped_mm_mxfp8` (cudnn) | `moe_gemm_mxfp8_nt_groupwise` (cute SM120) |
|---|---|---|
| SFA | 128x4-swizzled uint8, swizzled over the *global* packed `cum_m` (storage `round_up(cum_m, 128) x round_up(k/32, 4)`; per-expert scale segments share swizzle blocks) | DG-style **MN-major** per-token UE8M0 (int32-packed); per-expert start offsets aligned to 4 rows (≤3 pad rows per expert) |
| SFB | 3D 128x4-swizzled uint8, `(num_experts, n, k // 32)` | DG-style **MN-major** per-N-row UE8M0 (int32-packed), `(num_experts, n, k_align)` |
| Producing quantizer | `mxfp8_quantize(is_sf_swizzled_layout=True)` (TRT-LLM-style swizzle) | DG-style casts (`per_token_cast_to_fp8` + UE8M0 packing), following DeepGEMM conventions |
| K-axis granularity | 32 | 32 or 128 (64 can also be enabled — the kernel is GranK-templated) |

We deliberately aligned the cute entry with the **DeepGEMM ecosystem** (scale layout and
quant conventions), so callers already on DG-style scales can switch to this entry
without re-quantizing or re-laying-out scales.

So if the goal is folding this into `grouped_mm_mxfp8` as a `backend=`, the main design
question is the scale-factor contract of the unified entry: pick one layout (and repack
for the other backend internally), or accept both and dispatch on layout. We don't have
a strong position on this — what direction would you prefer?
