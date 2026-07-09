# Sub-task 0 findings: FP8 runner / thop 校验 contract (2026-07-09)

6KD 参照 @ `1e3557c`; FI 分支 @ `592f5ff`。

## FP8 runner moe 方法签名 (`cute_sm120_fp8_runner.h:54`)

```cpp
virtual void moe_gemm_fp8_nt_groupwise(
    void* D, void const* A, void const* B, int32_t const* token_offset,
    int num_groups, int max_shape_m, int shape_n, int shape_k,
    cudaStream_t stream, float const* SFA, float const* SFB,
    int scale_granularity_m = 1, int scale_granularity_n = 128,
    int scale_granularity_k = 128) = 0;
```

与 MXFP8 counterpart 差异 (有代码证据): scale 指针 `float const*` (vs `int32_t const*` packed);
尾参数 3 个 granularity (vs 单 `granK`)。其余逐位同构。

## thop 校验 contract (`thop/fp8GroupwiseGemm.cpp` moe path, 需完整 port 到 FI op)

| # | 检查 | 规则 |
|---|---|---|
| 1 | `check_inputs` | A/B: fp8_e4m3 + CUDA + contiguous; SFA/SFB: float32 + CUDA (**不查 contiguous**); 全部同 device |
| 2 | `check_layout(token_offset)` | int32 + CUDA + contiguous |
| 3 | dims | A `[M,K]` 2D; B `[E,N,K]` 3D; token_offset 1D size `E+1`; `A.K == B.K`; `E > 0` |
| 4 | `check_mnk` | `n, k > 0` 且 `% 16 == 0` |
| 5 | `check_scale_granularity_mnk` | 只接受 `(1, 128, 128)` |
| 6 | `check_zero_padding_sfa_layout` | shape 精确 `[Kb, MpE]`, `MpE = (M + 3E) // 4 * 4` (`compute_padded_offset`); contiguous; `data_ptr % 16 == 0` |
| 7 | SFB | `check_shape [E, Kb, Nb]` + contiguous |
| 8 | out | `empty [M, N]` bf16, A 的 device |

注: moe path 用的是 `check_zero_padding_sfa_layout` (contiguous 版), 不是 dense/batched 的
`check_sfa_layout` (stride 版, `aligned_m = ceil(M/4)*4` + backing storage 检查) — 不要 port 错。

## Test 侧 SFA 构造参照 (`test/test_fp8.py:140-161`)

```python
sf_a: per_token_cast_to_fp8(a) -> [M, Kb] K-major fp32
padded_sf_a = zeros(Kb, MpE)
expert i: padded_start = (token_offset[i] + 3*i) // 4 * 4
          padded_sf_a[:, padded_start : padded_start + m_i] = sf_a[start:end].T
SFB: per_block_cast_to_fp8(b) -> [E, Nb, Kb] -> transpose(-1,-2).contiguous() -> [E, Kb, Nb]
```

pad slot 留 0 即可 (kernel 不消费 pad 行)。6KD 自身 FP8 moe test 阈值 `calc_diff < 1e-3`,
cells 含 `E ∈ {4,8,16} × (num_rows, topk)` 随机 routing — FI task 用 uniform/uneven/empty cells,
另加一个 routing-style cell 对齐 6KD 覆盖面。

## Plan-vs-reality gap

- 无结构性 gap。一处命名修正: FFI launcher 函数名跟 MXFP8 先例 `CutlassMXFP8GroupwiseMoeGEMMSM120`
  对齐 → `CutlassFP8GroupwiseMoeGEMMSM120` (plan 里写的 `CuteFP8...` 弃用)。
- binding 显式传 3 个 granularity 参数 (不依赖 C++ 默认值), Python 侧 `scale_granularity_mnk`
  3-tuple 解包 — 与 MXFP8 binding 传参方式一致。
