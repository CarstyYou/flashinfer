# Shared Task Findings

记 important 发现 / bug / 踩坑 / 性能 等**客观现象**. **不记** subjective design decisions (xiy 自己的主观选择). 详细 artifact link 到 `task_NN/plan.md` / `task_NN/results/`. Hook (`mega_inference/.claude/hooks/task-findings-hook.sh`) enforce sediment 才允许 commit.

## task_01: 6KD kernel 目录整体搬运 + FI MXFP8 验证 (FP8 集成任务, 2026-07-08)

### Sync / 布局重构实测

- 16-file 整体搬运 (sm120_common 抽出 + FP8 family 进树不接线) 后, MXFP8 JIT module **无需改 JIT spec**
  即编译通过 (sources 只列 3 个 .cu, header 走 `extra_include_paths`); upstream test pre/post 均 69 passed。
- MXFP8 输出 pre/post **bit-identical** (40 cells calc_diff 逐 cell 完全相同) → 重构未改 MXFP8 数值行为。
- sync script 新增 quantize strip 规则 (runner.h 声明 / runner.cu include + 双函数, 尾部锚定 regex,
  命中数 != 1 即 abort) + `quantize_mxfp8_for_moe` 入 LEFTOVER_PATTERNS; 实跑残留 0。
- sync script 无 delete 能力, FI 侧被取代的 `sm120_blockscaled/{math,scheduler,utils}.cuh` 要手动 `git rm`。
- **sync 输出必须补 pre-commit (clang-format)**: FI in-tree 稳态是 clang-formatted (PR #3562 task_12 先例),
  6KD 原始风格不同 → 每次 sync 后 FI 侧跑 `pre-commit run --from-ref/--to-ref` 再 commit;
  sync script 的 byte-parity 只对 "sync 原始输出" 成立, 与 formatted in-tree 比较会有全量 whitespace diff
  (task_01 的大 churn 部分来自此)。FI pre-push hook 强制 lint-verified HEAD 才允许 push。
- **clang-format include-Regroup 会破坏 6KD header 编译** (实测 3ea8225 全 69 test 编译 fail):
  FI Google style = `IncludeBlocks: Regroup` 跨空行重排 include, 而 6KD header 的 include 顺序承载
  依赖 (headers 非 self-contained, 如 `sf_mxfp8_tma_load.cuh` 依赖先 include 的 `sm120_common/*`),
  重排后在 cutlass `cute/algorithm/copy.hpp` 报连锁错。修复: sync script `guard_include_block()`
  对每个 synced 文件的 prologue include 块注入 `// clang-format off/on`; 长期修法是 6KD headers
  self-contained 化 (6KD 侧 TODO)。

### 性能 (paired bench + 回退调查)

- 25/28 cells |delta| ≤ 0.7%, E=8 全部持平; E=4 小 M 3 个 cell 序列 bench 中出现 -1.5%~-4.5%。
- 回退调查定论 **kernel 零回退**: dispatch 源码一致 + SwapAB 路径 14 组件语义等价 (subagent) +
  SASS 逐指令一致 (4117 行) + NCU `sm__cycles_elapsed` 完全相等 (202072) + 隔离单 cell
  fresh-process bench 分布重叠。序列 bench 差异 = ~36us 小 kernel 的 run-to-run 方差。
  详见 [task_01/plan.md ## 回退调查](../task_01/plan.md)。
- 教训: 小 kernel (数十 us) 的 1-2us 级 bench delta 不能只靠"两轮一致"定为真实回退,
  必须 NCU cycles / SASS / 隔离进程三角验证 (6KD F9 的 FI 版)。

### NCU / 环境踩坑 (回退调查中发现)

- **NCU 注入下不能做 flashinfer JIT 编译**: ncu 的 TreeLauncher 注入所有子进程, ninja/nvcc
  工具链 hang (25 min 零编译进程, 全树 ~0% CPU)。正确流程: 先普通跑一次触发 JIT 编译,
  NCU 时 cache hit。表现为 bg shell 长时间无输出 (tail/管道缓冲), 需上节点 `ps` 判断。
- 容器 PID 1 不收割 zombie: kill -9 被 NCU hang 的进程树后残留 Z 态进程 (无资源占用);
  重启容器可清但会丢 /tmp 的 pip user-install (nvidia-cutlass-dsl 升级 + pytest) 与 JIT cache。

### 环境踩坑

- FI main (2026-07) `import flashinfer` 需要 `nvidia-cutlass-dsl >= 4.5.0`; 容器旧装 4.4.1 报
  `cutlass.cute.nvgpu` 缺 `OperandMajorMode`。已在 `xiyTrtllm` 容器升级到 4.6.0 (+装 pytest)。
- `pytorch:26.04-py3` 镜像无 pytest; pip user-install 落 `/tmp/.local` (容器重建即失)。
- pre/post paired bench 可用 `git stash` 往返切树; JIT 按 source SHA 自动重编 (~2 min/次), 无需手清 cache
  (正式对比仍建议清)。
- host (frontend) pre-commit 2.17.0 无法解析新式 hook manifest → FI commit 需容器内 lint 或 `--no-verify`。

## task_02: FP8 moe_gemm_fp8_nt_groupwise 接线 (2026-07-09)

### 实测

- FP8 op/binding/JIT/Python entry 全链路一次接通: validation 13/13, correctness 34/34
  (calc_diff 7.11e-4~7.14e-4, 与 6KD test_fp8 范围一致), bench vs cutlass grouped 24/24 全正
  (小 M E=4 +57~69%, E=8 +16~19%, m_pe=1024 +0.4~5.1%), 两轮无漂移。数据 link:
  [task_02/plan.md ## Results](../task_02/plan.md)。
- 单 .so 双 family: fp8 3 个 .cu 加入 `gen_gemm_sm120_module_cute_mxfp8` sources, 增量编译 1m20s,
  `.so` 1.02→1.52MB; 两个 binding 文件各自 `TVM_FFI_DLL_EXPORT_TYPED_FUNC` 共存无冲突
  (先例: cutlass gemm_sm120 module 双 binding)。
- **R3 复现**: `group_gemm_fp8_nt_groupwise` 在 2026-07 main 的 sm120 `num_groups > 1`
  wrapper-level disable 仍在 (`gemm_base.py:7283`); `skip_check=True` 可 bypass 作 perf baseline。

### 踩坑

- **validation test 用方阵 shape 会漏 SFB transpose 检测**: n=k 时 [E,Kb,Nb] transpose 后 shape
  不变, "bad shape" 用例静默通过 — negative test 的 shape 必须非对称 (Kb ≠ Nb)。
- FP8 thop 校验 port 时 review 抓到两处漏检 (`n/k > 0`、cross-tensor same-device): MXFP8 op
  不查这两项是因为 mxfp8 thop 自身没有, FP8 thop 有 → counterpart 对齐必须以各自 thop contract
  为准, 不能只 diff FI 侧文件。

## MXFP8 集成沉淀 (task_01–13, 2026-06, 已归档)

MXFP8 集成任务 (PR #3562) 的可复用 findings 摘编. 原 task_01..13 目录已清理; 完整 audit trail 见 `sm120_group_gemm_mxfp8_internal` 分支. 源码 line 引用基于 2026-06 main, 新 main 可能漂移.

### flashinfer JIT 踩坑

- **`FLASHINFER_JIT_VERBOSE=1` 隐式触发 debug mode** (`-G --device-debug -O0`, 编译时间 ~5×). 不想 debug 用 `FLASHINFER_JIT_DEBUG=0` 显式 release. 源码: `flashinfer/jit/core.py:415-417`.
- **flashinfer `3rdparty/cutlass` 默认未 init** (空 dir). 依赖 cutlass 的 JIT module 第一次 compile 报 `cutlass/cutlass.h: No such file`. 解决: `git submodule update --init --recursive 3rdparty/cutlass`.
- **flashinfer 现有 entry 可能有 sm120 disabled 分支**, bench 前必 grep (例: `group_gemm_fp8_nt_groupwise` sm120 曾显式 raise `RuntimeError("known correctness bug for num_groups > 1")`).
- **JIT module 单一化 pattern**: cutlass `gemm_sm120` module 把 dense + grouped source 全 bundle 进同一 JIT spec (单 `.so`); `gemm_sm120_cute_mxfp8` follow 同 pattern — 减少 compile time + cache 占用.
- **JIT spec 全 in-tree 后 `extra_include_paths` 可完全清空**: flashinfer JIT default include (`FLASHINFER_INCLUDE_DIR + FLASHINFER_CSRC_DIR + CUTLASS_INCLUDE_DIRS`, 见 `flashinfer/jit/cpp_ext.py`) 已覆盖所有 path 需求, 无需外部 repo env var.

### CUTLASS / SM120

- **`CUTLASS_ENABLE_GDC_FOR_SM100` 实际覆盖整 Blackwell family** (sm100/101/103/110/120/121), 不是 sm100 strict. CUTLASS macro 名取首发 arch, 不能据名推覆盖, 必须 grep source (`cutlass/arch/grid_dependency_control.h`).
- **flashinfer `3rdparty/cutlass` (b46b16d, v4.4.0+30) 跟 6KD bundled v4.4.2 API 兼容**. 6KD 用到的 cute/cutlass headers 全存在; 实测切换后 integration test 12/12 PASS, calc_diff 不变. → 上游 PR 不 vendor 6KD CUTLASS, 单一 source of truth.

### Binding 层约定 (6KD ↔ flashinfer)

- **Scale tensor 不能 `CHECK_CONTIGUOUS`**: transform 产出的 SFA 常是 contiguous storage `.transpose(0,1)` 后的非 contig view, kernel 读 raw pointer 按 native layout. binding 只 `CHECK_CUDA` + `CHECK_INPUT_TYPE`.
- **m_indptr ↔ token_offset 1:1 byte-equal**: flashinfer CSR cumsum int32 array 跟 6KD `int32_t* token_offset` byte 级一致, binding 层直接 pass-through 无 marshaling.
- **历史 note**: 6KD MGroupedContiguous 曾有 scheduler `get_expert_idx()` 恒 0 的 kernel bug (routing 全落 expert 0), 当时 workaround 是 contiguous entry 限 `use_psum_layout=true`. line refs 已过期, 复用 contiguous path 前需在当前 6KD source 重验.

### Bench baseline 约束 (sm120)

- **cutlass `group_gemm_fp8_nt_groupwise` csrc ICHECK 只接受 `(m,n,k) granularity ∈ {(1,128,128), (128,128,128)}`**. Python validator 不检, csrc 拦.
- **cutlass 该 entry 对 uneven m_pe 做 internal align-pad** (每 expert m_pe round 到 multiple-of-4, padded 行 zero), 接受 uneven m_indptr 不抛 SKIP.
- **`skip_check=True` 区分 wrapper-level vs hardware-level disable**: cutlass num_groups>1 sm120 disable 是 wrapper-level (bypass 后 dequant-self ref PASS); DG sm120 是 hardware-level (arch dispatch dict 写死 `{100a, 103a}`, KeyError, bypass 无效).
- **calc_diff 跨 entry 不可直比**: cutlass/DG UT 用 `dequantize → matmul` 自反 ref (calc_diff 极小); 6KD 集成 test 用 bf16 source ref (测 full quant pipeline, calc_diff ~7e-4). 两种 reference 测不同维度.

### 上游 test 约定

- **MXFP8 accuracy 用 `F.cosine_similarity > 0.99`** (UE8M0 block 量化 element-wise diff 偶超 `assert_close(atol=1e-2)`); 跟 flashinfer 已有 MXFP8 family test convention 一致. FP8 float-scale 不受此限, 仍用 calc_diff < 1e-3.
- **Zero_padding entry 的 test reference 需 per-expert 独立 quant**: sf 是 per-expert padded layout (per-expert 4-row pad + TMA align), 不是 token-contiguous; ref 用 per-expert PyTorch quant 独立算, 对 sf storage layout 不敏感的 metric (cos-sim / calc_diff) 才可比.
- **跨 repo byte-equal cross-check 是 testing-time dep**: 用 `sys.path` 注 6KD `test/utils/layout.py` 做 parity check 的 test 不上游, 上游 test 必须 100% flashinfer-only.

### Quant / layout helper 复用

- **DG `get_col_major_tma_aligned_packed_tensor` 是 arch-agnostic + granK-agnostic 的 SF transform building block**, 跟 6KD transform byte-equal; 优先复用 DG helper, 不 port 6KD 实现.
- **DG `transform_sf_into_required_layout` hardcode `arch ∈ {100a, 103a, 90a}` + granK=128**, sm120 silent fall-through; 改 existing 函数有 break risk → ship 平行 helper 而非 patch DG 函数.
