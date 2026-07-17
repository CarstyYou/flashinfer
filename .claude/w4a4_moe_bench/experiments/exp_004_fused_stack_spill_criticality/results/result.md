# exp_004：Fused Spill 问题点定位

## 结论

- **Register spill 是本项目的 P0 hard failure。** 该判定来自工程约束，不需要先证明 latency 影响。
- **两个物理问题点已定位：** 第一段 FC1 收尾阶段，`108 words/lane` 主块中的已完成 accumulator vectors 在各自 producer 后被逐步保存，并跨完整第二段 FC1 保活到 activation；`14 words/lane` 尾块在 activation 入口被保存，物理寄存器由 activation 临时值复用，随后再恢复。
- 主块是 first-pass accumulator；尾块不是同质 accumulator，而是 `5` 个 second-pass accumulator、`8` 个 index/address scalar 和 `1` 个 control scalar。
- **源码问题点尚未完全闭合：** 缺少 MLIR/PTX virtual value 到 SASS physical register/stack slot 的 compiler-certified 映射，9 个 scalar 也没有唯一 source SSA。
- **优化建议：无。** 在源码值与 allocator live interval 闭合前，任何 pass-order、lifetime 或调度改法都属于无证据推测。

## 1. 证据身份

| 项目 | 身份 |
|---|---|
| Kernel / launch | `MoEDynamicKernel`, grid `1×1×110`, block `160×1×1` |
| Baseline cubin | `9313fcbc0dd686f0684705e869fdd227608ac83ca43c1dc99d203f8e7143ca79` |
| Baseline SASS | `34b4c38161642a27ca6b4ec41ffad0bd70f6ff99fd8118997a4b2416c5e3abba` |
| Baseline NCU | `3367df71ef0c3b750c03c60436d100a994863271016995c76f21195dd9eaaea8` |
| Up-first cubin | `691ca03362e2e1efd7a2ad1af2d9074a38e5b809f7f0d99a6239e41f4009fee6` |
| Up-first SASS | `b7b236276e27acb2f7e0908f01df5e864ff2863d9a817aea67720026e55a6f98` |

## 2. Spill 发生在哪里

### 2.1 Main 108-word bundle

`27×OMMA output vectors → 54×STL.64 → 108 stack words → 108×LDL → 108/108 reloads first used by scale FMUL`

代表物理链：

```text
OMMA @0x7810 produces R12..R15
  → STL.64 @0x7860/@0x7880 saves stack[0x170..0x17c]
  → LDL @0xba90..0xbac0 restores R203..R200
  → gate scale/sigmoid @0xbce0..0xbd80
  → up scale @0xbd90
  → SwiGLU product @0xbda0
```

Store PC 范围 `0x7860..0x8060`；reload PC 范围 `0xba90..0xec30`。这把主 bundle 定位为第一段 FC1 收尾阶段随 producer 逐步保存、跨完整第二段 FC1 保活、在 activation 前后按需恢复的 first-pass accumulator。

### 2.2 Tail 14-word bundle

| Slot / reg | 原值 producer / 类别 | STL | activation 临时复用 | LDL 恢复 | 原值首个 consumer | 定位状态 |
|---|---|---:|---|---:|---|---|
| `0x1e4/R151` | OMMA R148..R151 @ `0xb9e0` / `second_pass_accumulator` | `0xbb40` | `0xbdc0` | `0xf050` | FMUL scale @ `0x11470` | physical + accumulator semantic closed |
| `0x1e0/R72` | OMMA R72..R75 @ `0xba60` / `second_pass_accumulator` | `0xbb60` | `0xbe20` | `0xf080` | FMUL scale @ `0x10540` | physical + accumulator semantic closed |
| `0x1dc/R73` | OMMA R72..R75 @ `0xba60` / `second_pass_accumulator` | `0xbb80` | `0xbe80` | `0xf090` | FMUL scale @ `0xff80` | physical + accumulator semantic closed |
| `0x1d8/R74` | OMMA R72..R75 @ `0xba60` / `second_pass_accumulator` | `0xbb90` | `0xbee0` | `0xf0b0` | FMUL scale @ `0xfb30` | physical + accumulator semantic closed |
| `0x1d4/R75` | OMMA R72..R75 @ `0xba60` / `second_pass_accumulator` | `0xbba0` | `0xbf40` | `0xf170` | FMUL scale @ `0x10d70` | physical + accumulator semantic closed |
| `0x1d0/R251` | SHF index @ `0x37c0` / `index_address_scalar` | `0xbbb0` | `0xbfa0` | `0xf260` | IMAD address @ `0x15140` | physical closed；unique source SSA unresolved |
| `0x1cc/R250` | SHF index @ `0x37d0` / `index_address_scalar` | `0xbbc0` | `0xc000` | `0xf2a0` | IMAD address @ `0x15150` | physical closed；unique source SSA unresolved |
| `0x1c8/R249` | SHF index @ `0x37e0` / `index_address_scalar` | `0xbbd0` | `0xc060` | `0xf2e0` | IMAD address @ `0x15160` | physical closed；unique source SSA unresolved |
| `0x1c4/R248` | SHF index @ `0x37f0` / `index_address_scalar` | `0xbbe0` | `0xc0c0` | `0xf320` | IMAD address @ `0x15170` | physical closed；unique source SSA unresolved |
| `0x1c0/R247` | SHF index @ `0x3800` / `index_address_scalar` | `0xbbf0` | `0xc140` | `0xf360` | IMAD address @ `0x15180` | physical closed；unique source SSA unresolved |
| `0x1bc/R246` | SHF index @ `0x3810` / `index_address_scalar` | `0xbc00` | `0xc1a0` | `0xf390` | IMAD address @ `0x15190` | physical closed；unique source SSA unresolved |
| `0x1b8/R245` | SHF index @ `0x3820` / `index_address_scalar` | `0xbc10` | `0xc240` | `0xf3b0` | IMAD address @ `0x151a0` | physical closed；unique source SSA unresolved |
| `0x1b4/R2` | LDG.E scalar @ `0xb650` / `long_lived_control_scalar` | `0xbc20` | `0xbca0→0xc2a0` | `0xf3c0` | FSETP control @ `0x120f0` | physical closed；unique source SSA unresolved |
| `0x1b0/R0` | IMAD index @ `0x37b0` / `index_address_scalar` | `0xbc30` | `0xbcb0→0xc300` | `0xf3d0` | IMAD address @ `0x151f0` | physical closed；unique source SSA unresolved |

`up_first` 保留全部 108-word 主 bundle，但消除全部 14 个 scalar save/restore；stack 从 `488 B/thread` 降到 `432 B/thread`，NCU local sectors 每方向精确减少 `568,064`。这支持 mixed activation-entry live set 机制，但不能把原因收窄成某一个源码构造。

## 3. 跨层闭合与边界

| 层级 | 已闭合 | 未闭合 |
|---|---|---|
| Python source | `gate_acc` 1651；`up_acc` 1652；Gate/Up GEMM 与 activation phase | source variable 到具体 physical spill slot |
| MLIR | `%rmem_118/%rmem_119` 的 alloc、GEMM、activation def-use（关键行 1015/1016/2726） | MLIR SSA 到 ptxas physical allocation |
| PTX | `%r2878/%r4955` 的 MMA def 与 activation use（关键行 2993/4773/5141） | PTX 无 spill local op 与 source location；virtual register 到 SASS register/slot |
| SASS | 108-word 全量 roundtrip；14-word 每项 producer→store→temporary reuse→reload→consumer | 9 个 scalar 的唯一源码 SSA |

因此当前状态是：**SASS physical mechanism-localized；source-value partially localized；实验尚未完全收口。**

## 4. 下一步取证

只补一类证据：同一编译身份下的 backend register-allocation/liveness dump，必须闭合：

```text
backend SSA/PTX virtual register
  → physical register
  → stack slot + spill/reload PC
  → producer/consumer live interval
  → source/MLIR location
```

单独增加 lineinfo 只能补 `PC→source line`，不足以闭合 value/register allocation。闭合前不进入优化设计。
