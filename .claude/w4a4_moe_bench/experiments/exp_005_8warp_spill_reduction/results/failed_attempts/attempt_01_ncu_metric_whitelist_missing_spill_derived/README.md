# Attempt 01：NCU custom metric whitelist 缺失 spill-derived 指标

`canonical_v0` 在 `--metrics` 中列出了四个
`sass__inst_executed_register_spilling_*` 指标，NCU 命令也成功返回，但两臂的
`native_raw.csv` 均未生成这些列。因此该 capture 不能证明动态 spill，禁止进入
正式结论。

根因是 compiler `SpillRefill` 分类依赖 `InstructionStats` 与
`SourceCounters` section；只写 metric ID 不足以触发该派生过程。

处理：

- 原始 `canonical_v0` 保留在本地，不覆盖、不混入正式证据；
- `canonical_v1` 使用九个 NCU sections 与非-spill custom metrics 的并集；
- capture 发布前解析 `native_raw.csv`，任一 required metric 缺失即删除临时目录并失败；
- capture identity 记录 section、custom metric 与 required metric 三套契约。

正式动态证据只来自 `canonical_v1`。
