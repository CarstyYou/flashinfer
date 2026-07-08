# Git 操作规则 (flashinfer 集成任务)

本文是本 repo 内 Git mutation policy 的唯一来源（吸收自 6KD `.claude/rules/git.md`）。
`workflow.md` 只引用本文，不复制 commit、merge、push 或回退规则。

## 操作前核对

- read-only Git 检查可直接执行；任何会改变 index、worktree、refs 或 remote 的操作前必须完整读取本文。
- mutation 前核对当前 branch、HEAD、`git status --short` 及预期 diff；实际 Git 状态优先于可能过期的文档。
- worktree 中已有改动默认属于 xiy；不得回退、删除、覆盖或混入当前任务。

## Branch 策略

- 在当前 branch 工作；除非 xiy 明确要求，不得创建 branch、tag 或 worktree。
- 双分支模式：`<task>_internal` 分支携带 `.claude/` agent infra + task 目录，完成迁移验证后由 xiy
  发起 strip 成公开分支（不含任何 `.claude/` xiy-specific 文件、task 目录）；strip 时机和方式由 xiy 决定。
- 将 task branch 合并回主线时默认 `git merge --no-ff`，保留 branch boundary；只有 xiy 明确指定
  squash、fast-forward 或 PR 时才改用对应方式；merge 授权不包含 push。

## Staging

- 按改动目的显式 stage 文件，禁止 `git add .` 或 `git add -A`。
- stage 前后检查 unstaged/staged 边界；commit 前必须展示并核对 staged 文件列表，不得混入无关、
  尚未 review 或不属于同一 owning purpose 的文件。
- mixed worktree 中只提交当前已授权范围，其他改动保持 unstaged。

## Commit

- 只有得到 xiy 明确授权时才 commit；授权只适用于当前已说明的改动范围。
- 除 Git 自动生成的 merge commit 外，commit subject 必须为 `[scope] <summary>`：
  - task 工作（binding / entry / tests / results / task plan）使用对应的 `[task_NN]`；
  - agent infra（`.claude/` rules / tasks 结构 / hooks 配套）使用 `[infra]`；
  - 纯文档或规则使用 `[doc]`；
  - 修复既有错误使用 `[fix]`；不改变对外行为的内部结构调整使用 `[refactor]`。
- task work file（tests/results/csrc/Python entry 等）改动必须同 commit staged shared
  `.claude/tasks/memory/findings.md`（hook enforce）；无 objective 新发现时 commit message 加
  `[no-findings]` marker bypass。
- 默认增量 commit；只有 xiy 明确要求时才 amend。
- commit 前核对拟用 subject 符合上述规则；commit 后检查新 HEAD 和剩余 worktree。

## Push

- push 必须单独得到 xiy 明确授权；commit、merge 或"完成任务"都不隐含 push 授权。
- 不 push upstream（flashinfer origin）；push 目标默认 xiy fork（remote `xiy`），且仍需明确授权。
- force-push 必须再次取得单独、明确授权。

## 回退与清理

- 失败 sub-task 只撤销当前 agent 在该 sub-task 中引入的候选改动，保留 xiy 和其他任务的 diff。
- 未经明确授权，不得使用 `git reset --hard`、`git checkout -- <path>`、`git restore <path>` 或
  `git clean` 等会丢弃工作区内容的命令。
- 优先用精确 patch 撤销自己的修改；只可直接删除本轮明确创建且确认无用户内容的临时文件。
- 回退后重新检查 `git status --short`，不得以"工作区 clean"为目标误删无关改动。
