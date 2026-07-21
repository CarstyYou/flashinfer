# exp_023 结果：Reject

第一快门已经失败：`13 warps / 416 threads` 的 role-only preflight 编译产物已产生
`72 B/thread` stack，SASS 中有 `18 STL + 61 LDL/LDL.LU`。当前 Opt 同 SHA 的既有证据为
`0 B/thread` stack。

| Gate A 证据 | 当前 Opt | Role preflight |
|---|---:|---:|
| Block | 288 threads | 416 threads |
| Stack | 0 B/thread | **72 B/thread** |
| SASS local store / load | — | **18 / 61** |

因此按预注册约束直接 **Reject**，没有继续实现 `M64xN128 x 2` 的 stage alias，也没有运行 IKET overlap 或 ABBA。这个结论只否定本实验锁定的设计束；它不证明 FC2/Scatter overlap 概念本身不可行。

完整静态证据见 [static_resources.json](static_resources.json)，原始摘取见
[resource.txt](raw/resource.txt) 和 [sass_local_instructions.txt](raw/sass_local_instructions.txt)，身份见
[manifest.json](manifest.json)。
