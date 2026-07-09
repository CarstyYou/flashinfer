# Task 03: FP8 entry PR-readiness (upstream test / docs / lint)

task_02 接线完成后的公开化收尾。strip 公开分支由 xiy 发起, 不在本 task。

## Scope

| Sub-task | 内容 | Gate |
|---|---|---|
| 1 | upstream test `tests/grouped_mm/test_cute_sm120_fp8.py` (mirror `test_cute_sm120_mxfp8.py` 结构; 100% flashinfer-only; calc_diff < 1e-3 + rejects-bad-input 参数化) | 容器 pytest 全 PASS |
| 2 | `docs/api/grouped_mm.rst` 加 FP8 section (mirror MXFP8 section) | autosummary 语法正确 |
| 3 | AOT 检查 | 无需改动 (`gen_gemm_sm120_module_cute_mxfp8` module 级注册已含 fp8 sources), 验证 aot.py 不再引用 per-entry |
| 4 | lint pass (容器 pre-commit --from-ref) + verified-head | pre-commit 全绿 |

## Risks

- upstream test 不得依赖 task_02 内部脚本 (self-contained helpers); MXFP8 test 先例可 import
  `flashinfer.testing.utils` (包内)。
- 新 py/cu 文件过 ruff/clang-format 可能 churn — lint 放最后统一跑。

## Results (2026-07-09)

| Sub-task | 结果 |
|---|---|
| upstream test | `tests/grouped_mm/test_cute_sm120_fp8.py` **24 passed** (2m04s, 含 JIT cache hit); 16 uniform cells + uneven/empty + 6 rejects 参数化 |
| docs | `grouped_mm.rst` 加 "FP8 MoE GEMM (SM120 cute backend)" section |
| AOT | 无需改动: `aot.py:582` 已按 module 注册 `gen_gemm_sm120_module_cute_mxfp8()`, fp8 sources 随 module 进 AOT |
| lint | 新文件 pre-commit 全绿 (ruff 自动重排 test) |

### Verdict

task_03 完成。FP8 entry PR-ready: kernel 树 (task_01) + 接线 (task_02) + test/docs (task_03)。
剩余: strip 公开分支 (xiy 发起) + PR body。
