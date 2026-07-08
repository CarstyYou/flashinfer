# Flashinfer 集成任务工作流

## 适用范围

把外部独立 kernel repo (如 `6KD_fp8_block_scale`) 集成进 flashinfer 的 binding 任务. 也适用任何"新增 entry / 重构现有 entry"类型 task. **不**适用 flashinfer 内部 kernel 调优 (上游 workflow 不在本文档).

## 硬件环境

- GPU: NVIDIA RTX PRO 6000 Blackwell (sm120a / sm120f)
- Docker: `gitlab-master.nvidia.com:5005/xiy/dev-image:trt-llm` (CUDA 12.8+ / 13.x)
- 父 repo: `/home/scratch.xiy_gpu/mega_inference/` (flashinfer 是 submodule)
- 外部 repo 也是 mega_inference 同级 submodule (例: `6KD_fp8_block_scale/`)

## 核心原则

1. **每个任务 / 实验独立目录** — `.claude/tasks/task_NN/` 收纳 plan + 测试结果, 不污染 repo root
2. **plan 先 lock + xiy review** — 接口签名 / scope / dispatch 矩阵在 commit 任何代码前 lock
3. **一次一个 sub-task**, 每步 build + verify 通过再进下一步 (no batch multi-step)
4. **数据 / 代码 grounding** — 没 source 或 测试数据不推测, grep 优先
5. **改完默认 stop unstaged** — xiy review diff 后 explicit 一句才 commit
6. **6KD / 外部 repo 内部不动** (Phase 1 集成 scope); binding 层做 translation
7. **汇报结论先行** — 每次汇报先给 high-level 结论 / 结果, 再展开证据和细节; 明确区分事实 / 假设 / 结论;
   一次只讨论一个决策, 不询问可从仓库自行确认的信息
8. **MXFP8 counterpart 对齐 (结构基线)** — 已有 MXFP8 对应文件时 (op `.cu` / jit binding / JIT spec /
   Python entry / test), FP8 新文件必须对齐其层次、词根和 codestyle, 除区分 family 所必需的后缀外不创建
   同义命名; 只有 FP8 contract 确有代码证据的差异 (float scale / granularity / dispatch) 才偏离, 并在
   plan / review 中说明
9. **Git mutation 规则唯一来源是 [git.md](git.md)** — staging / commit / merge / push / 回退全部照它执行

## Subagent 使用

- 层级上限固定为 1: 只有主 agent 可以创建 subagent; 任何 subagent 严禁再创建、调用或委派其他
  subagent. 该 hard constraint 不得由任务、skill 或并行需求放宽
- 边界清楚、上下文较重的调研任务 (探索 reference 实现 / 大面积 code survey / 跨文件 audit) 默认交
  subagent; 主 agent 负责限定问题、复核引用并整合结论, 最终设计判断不得外包
- code review 门禁见 Phase 3.5: reviewer 只报告 findings, 不修改文件

## 工作流 (6 阶段)

### Phase 1: 建立任务目录 + 讨论 plan/sketch

新任务起手:

```
.claude/tasks/task_NN/                 # 一个任务一个目录, xiy review 的单一入口
├── plan.md                              # 设计 + scope + sub-task 表 + risks + ## Results
├── sub_task_0_findings.md               # 外部 repo state ground truth (verify 后)
├── tests/                               # 任务的测试 / bench 脚本 (不放 flashinfer/tests/)
│   ├── smoke.py                         # 编译 / dispatch 冒烟
│   ├── test_correctness.py              # 精度 (calc_diff)
│   └── bench_<entry>_vs_<baseline>.py   # 性能 vs flashinfer 同类 entry
└── results/                             # 测试 / bench 产出落地
    ├── correctness_log.txt
    ├── bench_<entry>_vs_<baseline>.csv
    └── ...
```

**为什么 tests 放 task dir 不放 `flashinfer/tests/`**: task 是 review 单元, xiy 在一个目录里能看到 plan + tests + results 完整 audit trail; 任务结束 ship 给客户时 tests 可整体迁到 `flashinfer/tests/` 入 upstream.

**Findings 沉淀 dir**: 每 task 的 findings (只记 objective 发现 / bug / 踩坑 / 性能, 不记 subjective design decisions) 沉淀到 shared `.claude/tasks/memory/findings.md` 顶层 `## task_NN` section, **不**在 per-task dir 写 `findings.md`. 单一文件方便跨 task reference 互相引用 / cross-link 已落地条目; 避免 finding 散在多 dir 难维护. 详 Phase 6a.

`task_NN` 命名: 顺序编号 `task_NN` (e.g. `task_01`, `task_02`, ...), 跟 6KD `exp_NN` 风格一致. 任务的语义名 (e.g. "group_gemm_mxfp8 6KD cute 集成") 写在 `plan.md` 头部 + Scope 表里, 不入目录名 (避免 rename 麻烦 / 路径变长).

**plan.md 必含**:
- 任务目标 + scope (in / out)
- 外部 kernel repo 路径 + 当前 HEAD commit
- API 签名 sketch (Python entry signature + 各参数语义 + dispatch 矩阵)
- Sub-task 表 + per-task verification gate
- Risks 清单 (binding / JIT / compile / dispatch / dtype mismatch 等可预见点)

**讨论 + review**: plan 写完先给 xiy review (默认 stop unstaged). xiy 改 plan 后再 unlock 进 Phase 2. **不允许跳过 plan 直接写代码**.

可选: 复杂 / dispatch-多分支的设计 plan, 用 `codex exec` 二审做 Plan-vs-Reality alignment check (尤其当外部 repo state 跟 design 假设有 gap 时), verdict 写到 plan.md 底部.

### Phase 2: Sub-task 0 — Verify 外部 repo state

**永远先做**, design 假设 vs reality 的 gap 在这里 surface.

- Read 外部 repo: runner header (`*.h`) + runner cu (`*.cu`) + thop (`*.cpp`) — 列 entry point + signature + GemmType / dispatch enum
- 写 finding 到 `tasks/task_NN/sub_task_0_findings.md`:
  - Runner methods 表 (signature + 内部 enum + thop 暴露状态)
  - Tile / kernel 分支阈值 (e.g. `m_per_expert` regions)
  - 跟 plan.md 描述的 reality gap (如有, 必须明列)
- xiy review finding 后决策 path forward (例: scope 收窄 / 扩外部 repo / re-investigate 别的分支)

### Phase 3: 实施修改 (sub-task 1 → N 顺序, gate-by-gate)

按 plan.md 列的 sub-task 表顺序推进. 典型集成任务的 sub-task 模板:

| Sub-task | 目标 file | Gate |
|---|---|---|
| 1. C++ binding skeleton | `csrc/<entry>_<chip>.cu` + `csrc/<entry>_<chip>_jit_binding.cu` | AST parse OK + 文件存在 |
| 2. JIT module spec | `flashinfer/jit/gemm/core.py` (新 `gen_*_module()`) + `jit/gemm/__init__.py` 导出 | `import flashinfer.<entry>` 不报错 |
| 3. Python entry | `flashinfer/gemm/gemm_base.py` 新 entry + accessor + `gemm/__init__.py` 导出 | validation tests (参数矛盾 → ValueError; 未实现 path → NotImplementedError) PASS |
| (parallel) Quantize helper | 同 csrc launcher / jit_binding + Python entry | bit-equal cross-check vs 外部 repo Python ref |
| First compile + smoke test | (运行) | smoke kernel call 出 tensor (shape + dtype) 不 crash |
| 4. Correctness test | `tasks/task_NN/tests/test_correctness.py` | calc_diff < 1e-3 (or task-specific threshold) |
| 5. Perf bench | `tasks/task_NN/tests/bench_<entry>_vs_<baseline>.py` | 全 cells ≥ baseline OR documented gap |
| 6. Doc | docstring inline + 必要时 `.rst` | `help(<entry>)` 渲染 OK |

**改完每个 sub-task 默认 stop unstaged**, 让 xiy review diff. 不主动 commit. 不主动 batch 多 sub-task.

Pattern 参考:
- flashinfer 现有 csrc/* (e.g. `csrc/group_gemm_fp8_groupwise_sm120.cu` + `csrc/group_gemm_sm120_binding.cu`)
- `flashinfer/.claude/skills/add-cuda-kernel/SKILL.md` (canonical FlashInfer kernel addition walkthrough)
- `flashinfer/CLAUDE.md` JIT 章节 (FLASHINFER_GEN_SRC_DIR / FLASHINFER_CSRC_DIR 规则)

### Phase 3.5: Subagent code review (每轮 code 改动后强制)

每轮 binding / entry / test code 修改完成后、build/test/commit 前, 主 agent 创建 subagent 依据当前
`task_NN/plan.md` review 实际 code diff:

- reviewer 只报告 findings (plan alignment / correctness / corner case / codestyle / dead code), 不修改文件
- 存在 MXFP8 counterpart 文件时, review 必须逐项对照 naming、词根、层次和 codestyle, 不能只审逻辑正确性;
  有意差异必须明确解释
- 主 agent 修复 finding 后只要再次改动 code, 就必须重新触发 review, 直到最近一轮没有阻塞 finding
- 纯文档 / 规则修改不触发该门禁

### Phase 4: Build + Verify

**4a. JIT compile**

- 容器内: `pip install --no-build-isolation -e .` (editable install)
- 触发 JIT: 首次 kernel call (lazy compile)
- Release mode 必设: `FLASHINFER_JIT_DEBUG=0 MAX_JOBS=8 FLASHINFER_NVCC_THREADS=2`
- 不设 → 走 legacy `FLASHINFER_JIT_VERBOSE=1` 隐式触发 debug mode (`-G --device-debug -O0`, ~5× 慢)
- Compile fail 常见原因:
  - `cutlass/*.h: No such file` → flashinfer 自己的 `3rdparty/cutlass` 未 init OR 跟外部 repo CUTLASS 版本不兼容 → 在 JIT spec 加外部 repo CUTLASS path via `extra_include_paths` (`-I` 优先级高于 flashinfer 的 `-isystem`)
  - 外部 kernel header path 缺 → 检查 `extra_include_paths` 是否含外部 repo include dir
- JIT cache 清: `rm -rf ~/.cache/flashinfer/<version>/<arch>/cached_ops/<module_name>/`

**4b. 正确性测试** — Phase 3 sub-task 4 跑出 calc_diff

- Metric: `calc_diff(out, ref) = 1 - 2<x,y>/(||x||² + ||y||²)` (跟外部 6KD repo `benchmark.py:calc_diff` 一致)
- Threshold: `< 1e-3` baseline (跟 6KD `test_moe_gemm` 一致); 调严需 explicit 一句
- Reference: bf16 PyTorch matmul on dequantized inputs (per-expert / per-group loop)
- 失败处理: `git checkout` revert 改动 → 检查 dispatch 路径 / quant 层 / TMA descriptor → 修复后重测

**4c. 性能测试** — Phase 3 sub-task 5 跑出 us + speedup

- Baseline: flashinfer 现有同类 entry (例: `group_gemm_mxfp8_nt_groupwise_zero_padding` vs `group_gemm_fp8_nt_groupwise`). **不 vs DG / 不 vs 外部 repo direct thop**, 除非 explicit binding overhead 测量
- Timing: warmup 10 iter + 50 iter median (`torch.cuda.Event`)
- 比 ratio 用百分比 (`(t_base - t_ours) / t_base * 100`), **不用倍率** 除非 ≥ 2×
- 跨 cells (E / m_pe / N / K) 跑表; problem size 要满足 baseline constraint (例: fp8 entry 要求 m_pe % 4 == 0)
- 注意 flashinfer baseline 在 sm120 上可能有 disabled path (例: `num_groups > 1` correctness bug → `gemm_base.py:7006-7010`), 只能在 supported cells 对比

**4d. 验证纪律**

- 没有当前输出或具体日志证据时, 不得声称 build / test 已通过
- smoke、临时诊断、probe 的日志 / CSV 固定写节点或 container 的 `/tmp`, 不放进 task 目录; 只有正式
  correctness gate 和正式 bench 结果才写入 `task_NN/results/`

### Phase 5: 记录精度 + 性能

落到 `tasks/task_NN/results/` 下, 也复制结论到 `tasks/task_NN/plan.md` 底部 `## Results` section.

**精度结果** (`results/correctness_<cell>.txt` 或 plan.md 表格):
- 每 cell: `calc_diff` 值 + threshold + verdict
- bit-equal cross-check 单独记 (quant helper vs 外部 ref)

**性能结果** (`results/bench_<entry>_vs_<baseline>.csv`):
- CSV header: `E,m_pe,N,K,<ours>_us,<baseline>_us,speedup_pct`
- 表格 markdown 复制到 plan.md, 加 cells 间观察 bullet (不超过 5 句 prose)
- 异常值 (cell 速度反常) 必须解释 (例: 命中 baseline tile boundary / 命中 dispatch corner case)

**`plan.md ## Results` section template**:

```markdown
## Results (YYYY-MM-DD)

### Correctness
| cell | calc_diff | threshold | verdict |
|---|---|---|---|
| ... | ... | < 1e-3 | ✓ PASS |

### Perf vs <baseline>
| E | m_pe | N | K | ours (us) | baseline (us) | speedup |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | +X.X% |

### 观察
- <最大值 cell + reasoning>
- <最小值 cell + reasoning>
- <known anomaly + dispatch boundary 解释>

### Verdict
M{1/2/3/4} milestone reached. <剩余 sub-task / 待办 / Phase 2 scope>.
```

### Phase 6: Findings 沉淀 + Commit + 收尾

**6a. Findings 沉淀** — append 到 shared `.claude/tasks/memory/findings.md` 顶层 `## task_NN: <一行任务名>` section, **不**在 per-task dir 写 `findings.md`. 风格: bullet + link, **只记 objective 发现 / bug / 踩坑 / 性能 等客观现象, 不记 subjective design decisions** (e.g. naming convention / scope choice / version pivot / task status summary). 详细 artifact (cell-level 数据 / report) link 到 `task_NN/plan.md` 或 `task_NN/results/`. 典型 `###` 子 section (全 objective):

- **<area> 踩坑** — 库 / JIT / 编译 / 接口 layout 等 hidden constraint 实测 (e.g. flashinfer JIT debug-mode 隐式 trigger / CUTLASS version mismatch / scale tensor 非 contig)
- **<topic> 实测** — validation / correctness / bit-equal / perf 等 sub-task 结果 (cell-level 数据 + 数值 link 到 results/)
- **性能** — bench numbers + root cause 分析 (e.g. uneven m_pe regression -3.3% w/ tile dispatch heuristic 假设)
- **风险命中 / 不命中** — plan.md ## Risks 表 close, with objective evidence
- **跨 entry / 跨 repo 对比** — 不同 entry 在同 input 上的实际行为差异 (e.g. cutlass DISPATCH ICHECK enforcement, calc_diff 跨 entry 不可直比 reasons)
- **Kernel-level bug 诊断** — 源码 line 引用 + 触发 condition + workaround / fix 候选 (e.g. MGroupedContiguous get_expert_idx routing bug)

每条 finding 一行 bullet, 必要时 sub-bullet 列证据 / 源码 line 引用. 没 objective 新发现可以 skip section (用 commit message `[no-findings]` marker), 不强制凑数. Hook `mega_inference/.claude/hooks/task-findings-hook.sh` enforce: task work file (tests/* + results/* + csrc/*.cu + Python entry 等) 改动 必须 同 commit staged shared findings.md, 否则 block. Bypass: commit message 加 `[no-findings]` marker (plan-only / trivial fix / WIP / no objective finding).

**6b. Commit**

- xiy review `plan.md ## Results` + shared `tasks/memory/findings.md` `## task_NN` section + 各 sub-task diff 后 explicit 一句 → commit
- staging / commit subject / merge / push / 回退规则全部遵循 [git.md](git.md), 本文不复制
- 涉及多 repo (mega_inference 父 + flashinfer submodule + 外部 repo) 时, 分 repo commit; 父 repo 只在
  submodule main advance 时 bump pointer
- 不主动 push / tag

## 不做 / 不主动

- 不改 flashinfer 现有 csrc / module / Python entry (除非 explicit, 例: hook 现有 dispatch 时)
- 不改外部 repo 内部 (Phase 1 scope; Phase 2 等客户验收)
- 不主动 commit / push / tag (per memory `[[feedback_user_review_before_commit_push]]` + `[[feedback_no_auto_tag_unless_ship]]`)
- 不 delete files (per `[[feedback_no_delete]]`)
- 不 batch 多 sub-task 一次 verify (per `[[feedback_incremental_verify]]`)
- 不在 kernel / binding 内加 inline 注释 (除非 explicit 一句, per `[[feedback_no_comments_unless_asked]]`); 1-line // 解释 hidden constraint OK

## 关键 paths

| Symbol | Path |
|---|---|
| `$MEGA_ROOT` | `/home/scratch.xiy_gpu/mega_inference` |
| `$FI_ROOT` | `$MEGA_ROOT/flashinfer` |
| `$EXTERNAL_REPO` | `$MEGA_ROOT/<repo_name>` (sibling submodule, e.g. `6KD_fp8_block_scale`) |
| `$TASK_DIR` | `$FI_ROOT/.claude/tasks/task_NN/` |
| `$FI_CSRC` | `$FI_ROOT/csrc/` |
| `$FI_JIT` | `$FI_ROOT/flashinfer/jit/gemm/core.py` |
| `$FI_PY` | `$FI_ROOT/flashinfer/gemm/gemm_base.py` |
| `$FI_TESTS` | `$FI_ROOT/tests/gemm/` |
| `$JIT_CACHE` | `~/.cache/flashinfer/<ver>/<arch>/cached_ops/<module>/` (container 内) |

## 常用命令

容器 (host shell):
```bash
just docker-up                   # 起容器 (mega_inference 父 repo Justfile)
just docker-id                   # 拿 container ID
```

容器内 flashinfer dev:
```bash
docker exec -w $FI_ROOT $CID bash -c "pip install --no-build-isolation -e . -v"
docker exec -w $FI_ROOT $CID bash -c "FLASHINFER_JIT_DEBUG=0 MAX_JOBS=8 FLASHINFER_NVCC_THREADS=2 python3 .claude/tasks/task_NN/tests/<test>.py"
docker exec $CID bash -c "rm -rf ~/.cache/flashinfer"   # JIT cache clear
```

环境变量:
- `FLASHINFER_JIT_DEBUG=0` — release mode (跟 `=1` 差 ~5× 编译时间)
- `MAX_JOBS=N FLASHINFER_NVCC_THREADS=M` — parallel JIT
- `FLASHINFER_<EXTERNAL>_ROOT=<path>` — 外部 repo 路径 override (default = sibling submodule)
- `FLASHINFER_JIT_LINEINFO=1` — release + ncu source 关联 (不带 `-G`)

## 关联 docs / skills

- `$FI_ROOT/CLAUDE.md` — flashinfer JIT 架构 + cache 规则 + 全部 env var
- `$FI_ROOT/.claude/skills/add-cuda-kernel/SKILL.md` — flashinfer 新 kernel canonical 步骤
- `$FI_ROOT/.claude/skills/benchmark-kernel/SKILL.md` — CUPTI / bench_gpu_time 用法
- `$FI_ROOT/.claude/skills/debug-cuda-crash/SKILL.md` — `@flashinfer_api` API logging 调试 CUDA crash
- `$MEGA_ROOT/CLAUDE.md` + `$MEGA_ROOT/CLAUDE.local.md` — 父 repo 集成约定
