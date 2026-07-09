# Sub-task 0 findings: 外部 repo state verify (2026-07-08)

6KD HEAD `5c562b1` / FI 分支 `sm120_moe_gemm_fp8_internal` @ `a25af45` + infra commit `2c57ad1`。

## sync --dry-run 结果

16-file FILE_MAP 与 plan 文件树完全一致: 5 overwrite (mxfp8 runner.h/.cu + sm120_blockscaled 3 个) +
11 create (fp8 runner.h/.cu + sf_mxfp8_tma_load + sm120_blockscaling 4 个 + sm120_common 4 个)。
script 无 delete 能力 → FI 侧 `sm120_blockscaled/{math,scheduler,utils}.cuh` 需手动 `git rm` (план DEL 项)。

## include 闭包

全部 sync 文件的 local include 在 FILE_MAP 目标集内闭合, 唯一 offender:
`cute_sm120_mxfp8_runner.cu:26` → `sm120_blockscaled/quantize_mxfp8_for_moe.cuh` (R1, 见下)。
6KD source 无任何对已删除 `sm120_blockscaled/{math,scheduler,utils}.cuh` 路径的引用。

## R1: quantize_mxfp8_for_moe 足迹 (不参与集成, xiy 指定; 方案 = sync script 加 strip 规则)

| 位置 | 内容 | strip 规则 |
|---|---|---|
| `cute_sm120_mxfp8_runner.h` 尾部 (class 外, namespace close 前) | free-fn 声明 `void quantize_mxfp8_for_moe(...int granK = 128);` | 尾部锚定 regex 删到 namespace close |
| `cute_sm120_mxfp8_runner.cu:26` | `#include ".../quantize_mxfp8_for_moe.cuh"` | 精确行删除 |
| `cute_sm120_mxfp8_runner.cu` 尾部 | `quantize_mxfp8_for_moe_impl<GranK>` static fn + public `quantize_mxfp8_for_moe` fn (连续两函数, namespace close 前) | 尾部锚定 regex 删到 namespace close |

强制校验: `quantize_mxfp8_for_moe` 加入 LEFTOVER_PATTERNS — 任何残留 (含未来 drift 引入的新引用)
直接 fail sync; strip 规则命中数不符也 hard error。

## FI op.cu ↔ 6KD 当前 runner 接口

- `moe_gemm_mxfp8_nt_groupwise(D, A, B, token_offset, num_groups, max_shape_m, shape_n, shape_k, stream, SFA, SFB, granK)` — FI op.cu:118 调用参数逐位匹配 ✓
- op.cu 只 include `cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.h` + `tvm_ffi_utils.h`, 不受布局重构影响 ✓

## JIT module 影响

`gen_gemm_sm120_module_cute_mxfp8` sources 只列 3 个 .cu (runner/op/binding), header 走
`extra_include_paths=[FLASHINFER_CSRC_DIR]` → 布局重构 **不需要改 JIT spec**; 仅 docstring 提及
`sm120_blockscaled/` 目录 (cosmetic, 顺手更新)。JIT cache 按 source SHA 失效, sync 后 runner.cu 变
→ 自动重编, 但为排除半失效仍显式清 cache (plan R4)。

## MOD 文件 diff 规模 (post-SUBS vs FI in-tree 现状)

| 文件 | 6KD 行数 | FI 旧行数 | changed lines |
|---|---:|---:|---:|
| cute_sm120_mxfp8_runner.h | 152 | ~168 | 165 |
| cute_sm120_mxfp8_runner.cu | 453 | ~527 | 522 |
| sm120_blockscaled/builder.cuh | 152 | 596 | 572 |
| sm120_blockscaled/kernel_impl.cuh | 511 | 469 | 432 |
| sm120_blockscaled/launch.cuh | 169 | 134 | 139 |

实质是整体重写 (exp_19..32 + sm120_common 抽取), 不是增量 patch → plan R3 成立,
MXFP8 correctness + perf 必须全量重验, 回退时 bisect 无意义, 直接按 shared 文件粒度排查。
