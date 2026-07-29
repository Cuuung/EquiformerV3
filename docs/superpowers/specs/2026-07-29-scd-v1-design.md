# SCD v1 设计：AdaNorm 条件注入

分支：`scd-v1`（自 `scd-v0` 派生，`compile-equiformer-v3` 是其祖先）

参考文献：Perez & Gómez-Bombarelli, *Self-Conditioned Denoising for Atomistic
Representation Learning*（2026），仓库内副本 `repositories/scd/docs/scd.pdf`。
参考实现：`repositories/scd`（TorchMD-Net 主干）。

## 1. 背景与目标

v0 已上线并验证（24 项测试全过），采用 DeNS 同款的**输入层加法**注入条件，
换取零侵入：不改 `TransBlockV3` / `core_compute` 签名，完整复用 DeNS 的
direct / gradient / 两条编译路径。

v1 实现论文形态：把条件经 **AdaNorm** 注入每个 transformer block 的 pre-norm
（论文 §3.3、附录 B："we replace existing pre-attention layer normalization with
Adaptive Layer Normalization (AdaNorm), analogous to its use in Diffusion
Transformers"）。

目标：
1. AdaNorm 条件注入，与 v0 的输入层注入并存、可切换、可消融。
2. 元素嵌入冻结的四档开关。
3. 按论文确定正则化策略与强度，留出参数接口。
4. 纯 fp32 eager 的 N2L2C64 配置，先验证正确性。

**非目标**：性能优化、无标签 SSL 预训练的数据侧支持、微调阶段的 reset-head 协议。

## 2. 架构

### 2.1 单一模型类 + 注入模式开关

沿用 `equiformer_v3_scd`（不新建 v1 类）。新增 `scd_inject: input | adanorm | both`，
默认 `adanorm`。条件向量的生产链路完全复用 v0 已验证的代码：

```
pos_clean ──建图──> core_compute ──> x_scalar ──sum-pool──> ProjHead ──> c [B, C]
                                                                          │
                              dropcond(p=0.2) / mask token / clamp ───────┤
                                                                          ▼
   inject=input   ──> 写 L=0，叠加到 force_embedding      （v0 路径，已测）
   inject=adanorm ──> c[batch] 透传给每个 TransBlockV3    （v1 新增）
   inject=both    ──> 两者同时
```

理由：clean 前向、DropCond、ProjHead、编译路径兼容这些与注入方式无关，v0 的
测试覆盖可直接继承；三种模式并存使 "输入层注入 vs AdaNorm" 成为一次直接消融。

**`scd_inject` 只控制 SCD 自条件这一条路径。** DeNS 的真实力条件由独立的
`use_force_cond` 控制（v0 已有），两者正交：`scd_inject=adanorm` +
`use_force_cond=true` 表示力条件仍走输入层加法、自条件走 AdaNorm。

### 2.2 `cond` 的透传方式

**方案 A（采纳）：改签名显式透传。**

- `TransBlockV3.forward(..., cond=None)`
- `EquiformerV3_OC._forward_blocks(..., cond=None)`
- `EquiformerV3_OC.core_compute(..., cond=None)` 与 DeNS 版 `core_compute(..., force_embedding, cond=None)`

默认 `None` 保证不传 cond 时逐位等价于现有代码。

已否决的方案：

- **B. 模块状态注入**（`block.set_cond(c)`，签名不变）：`_conservative_compiled_forward`
  的文档注释明确要求每个 per-batch 张量必须是 traced region 的显式入参而非闭包捕获，
  B 直接违反，会 stale-bake。
- **C. 搭车进 `x` 的空闲通道**：`x` 无空闲通道，且污染特征张量。

### 2.3 新模块 `EquivariantAdaNorm`

位置：`experimental/models/equiformer_v3/layer_norm.py`

```python
x = self.norm(x)                                   # 复用 get_normalization_layer，[N, (L+1)^2, C]
inp = cat([cond, x[:, 0, :].detach()], dim=-1)     # 对齐参考实现 adaLN2
shift, scale, gate = self.fc(inp).split(...)       # fc 末层 zero-init
x = x * (1 + scale[:, expand_index, :])
x = x + pad_to_l0(shift)                           # out-of-place
return x, 1 + gate[:, expand_index, :]
```

**调制范围：per-degree。** `shift` 形状 `[N, C]` 只加到 L=0；`scale` / `gate`
形状 `[N, lmax+1, C]`，经 `expand_index` 广播到 `(lmax+1)²`。

`expand_index` 复用 `EquivariantMergeLayerNorm` 已有的 buffer 模式。

**等变性论证**：`self.norm` 等变（现有）；`scale` / `gate` 是不变量（源自 `cond`
与 `x` 的 L=0 分量），且同一 `(l, c)` 的全部 `2l+1` 个 m 分量乘同一个数，故等变；
`shift` 只加在 L=0，故等变。**同一 `(l, c)` 内 m 共享同一标量是等变性的充要条件**，
`expand_index` 就是保证这一点的机制。

选择 per-degree 而非全 degree 共享：`lmax=4` 时各 degree 的特征尺度差异大，
共享单一 scale 会失配；且 per-degree 与 `EquivariantMergeLayerNorm.affine_weight`
的形状 `(lmax+1, C)` 同构。

`scd_adanorm_scope: per_degree | shared | l0_only` 可退化成另两种做消融
（`l0_only` 对应参考实现 `dx*gate_x, dvec` 的严格形态）。

### 2.4 两处必须点名的偏离

**(1) `gate` 用 identity-init，不是 DiT 的 zero-init。**

DiT 的 adaLN-Zero 令 `gate = 0`，残差支在 step 0 完全关闭 —— 那是从零训练的设定。
本仓库需要从现有 equiv3 / DeNS ckpt 续训，`gate = 0` 等于把预训练 backbone 整个关掉。

因此：`fc` 末层仍 zero-init，但读出时用 `1 + gate`、`1 + scale`，`shift` 加 0。
初始时 AdaNorm 逐位等价于原 norm，**整个模型逐位等价于现有 equiv3**。写成测试断言。

**(2) `shift` 用 out-of-place。**

v0 中 `cond_embedding[:, 0, :] = c` 是对全新 zeros 张量赋值，安全。此处若对 `norm`
的输出做 in-place index_put，会引入 version-counter 与 make_fx 的问题。改为构造一个
只有 L=0 非零的张量再相加。

### 2.5 注入位置：`norm_1` 与 `norm_2` 都换

`TransBlockV3` 有两个 norm（`norm_1` 前置 attention，`norm_2` 前置 FFN），而参考实现
的 ET block 只有一个（FFN 融进了 attention 的 `o_proj`）。论文原话 "analogous to its
use in Diffusion Transformers" 指向 DiT 的标准做法 —— DiT 对 attention 和 MLP 各给一组
shift/scale/gate。ET 只有一个 norm 是其 block 结构使然，不是有意的限制。

```python
x_res = x
x, gate_a = adanorm_1(x, cond)
x = ga(x) * gate_a
x = drop_path(x) + x_res

x_res = x
x, gate_f = adanorm_2(x, cond)
x = ffn(x) * gate_f
x = drop_path(x) + x_res
```

`scd_adanorm_targets: ['attn', 'ffn']`（默认）/ `['attn']`（消融）。未列入的位置退回
普通 `get_normalization_layer`，行为与现有代码逐位一致。

每个 block 拥有独立的调制 MLP（对齐 DiT 与参考实现的 `conditional_ln` per-layer）。
成本：N@7 L@4 C@128 约 +2.5M 参数。

### 2.6 三条前向路径的改动

- `_forward_direct`：`plain_compile(self.core_compute)` 自动把新形参当输入，无需额外处理。
- `_forward_gradient`：直接调 `core_compute`，加实参即可。
- `_conservative_compiled_forward`：`core_fn_stress` / `core_fn_force` 需把 `cond`
  像现有的 `fe` 一样加进显式入参列表与 `dynamic_dims`。

clean 前向仍走**未编译**的 `core_compute`（v0 已验证），避免为 clean 图的形状触发第二轮编译；
它以 `cond=None` 调用 —— clean 前向本身是无条件的，条件正是由它产生。

## 3. 元素嵌入冻结（四档）

`scd_freeze_element_embedding: none | sphere | sphere_edge | all`

| 档位 | 冻结对象 | N@7L@4C@128 | N2L2C64 |
|---|---|---|---|
| `none` | — | 0 | 0 |
| `sphere` | `sphere_embedding` | 16,384 | 8,192 |
| `sphere_edge` | ↑ + `edge_degree_embedding.{source,target}_embedding` | 49,152 | 40,960 |
| `all` | ↑ + 每个 `blocks[i].ga.{source,target}_embedding` | 278,528 | 106,496 |

**明确排除输出头。** `force_block` / `dens_block` / `stress_block` 也是
`EquivariantGraphAttention`，各带一对 `source/target_embedding`，但它们是任务头而非
输入通道 —— SSL 预训练时 `dens_block` 正是被训练的那个头。论文关注的是输入侧元素
嵌入塌缩（附录 B："Without freezing element embeddings, we find that element embeddings
become vanishingly small, leading to downstream instability"）。

同时提供 `scd_freeze_mask_token: bool`（参考实现 `freeze_embeddings` 一并冻结
`mask_token`）。

**优化器与 DDP 安全性（已核实）**：`muon.py:432` 的 `if not p.requires_grad: continue`
与 `base_trainer.py:774` 都会跳过冻结参数；HybridMuon 仅在组非空时建组。冻结参数不进
autograd 图，不触发 DDP unused-param。

**为什么默认 `none`**：equiv3 的元素身份有三个入口共 `1 + 2 + 2×num_layers` 张表
（`sphere_embedding` 1 张、`EdgeDegreeEmbedding` 2 张、每个 attention block 2 张），
N@7 为 17 张、N2L2C64 为 7 张，而参考实现的 ET 只有 1 张。
论文结论不能直接外推 —— 只冻 `sphere_embedding` 时元素信息仍可从另两个入口进入，
塌缩未必被阻止，甚至可能只是转移到其余表上。

**配套诊断埋点**：在 trainer 中周期性记录三组元素嵌入的权重范数
（`sphere` / `edge_degree` / `blocks` 均值）到 wandb。一次 `none` 档预训练即可回答
"要不要冻、冻到哪档"，避免盲试四档。

## 4. 正则化策略与强度

| 项 | 论文（Table 17） | repo 现状 | v1 默认 | 接口 |
|---|---|---|---|---|
| 条件 dropout | 0.2 | v0 已有 | 0.2 | `scd_p_dropcond` |
| DropPath | 0.1 | 0.05 | **0.1** | `drop_path_rate` |
| clean 前向 reg noise σ | 0.005 | 无 | **0.0** | `scd_reg_noise_std`（新） |
| 腐蚀 noise σ | 0.04 | 0.025 | 0.025 不动 | `denoising_pos_params.std` |
| weight decay | 0.05 | 1e-3 | 1e-3 不动 | 已有 |
| 条件 clamp | ±100 | v0 已有 | 100 | `scd_cond_clip` |
| 冻结元素嵌入 | 冻（单表） | 无 | `none` | `scd_freeze_element_embedding` |

**DropPath 0.05 → 0.1**：论文把这条直接系在双前向上（附录 B："Because SCD pretraining
requires two forward passes per-step we find that including drop path with a drop
probability of 0.1 improves training stability"）。v1 正是双前向。注意 equiv3 的
`GraphDropPath` 是 **per-graph** 丢、参考实现的 `JointDropPath` 是 **per-node** 丢，
同一数值下 equiv3 的正则更强 —— **列为第一个待调项**，不是照抄即可。

（equiv3 的 irreps 是单一张量、degree 为中间维，掩码广播必然覆盖所有 degree，
因此论文要求的"联合丢弃 L0/L1"在 equiv3 中天然满足。）

**reg noise 默认 0**：参考实现只在 `noise_in_loader=False` 分支施加 reg noise
（`trainer.py:486-488`），而周期材料配置 `pretrain_amp20.yaml` 是
`noise_in_loader: True` —— 作者自己在材料上就没走这条分支。本项目场景是材料，
无正面证据，留开关但不默认开启。

**weight decay 与腐蚀噪声不动**：repo 这两个值在本地数据上调过，论文值是 QM9/PCQ
尺度，改过来是回退而非改进。

## 5. 测试

扩展 `experimental/tests/test_equiformer_v3_scd.py`（现有 24 项之上）：

1. **identity-init 逐位一致**：把 `scd_inject=adanorm` 模型的公共权重原样加载进
   `equiformer_v3_dens`（同 seed、同输入、同 batch），两者输出必须逐位相同 ——
   即调制头未经训练时 AdaNorm 恒等于原 norm。验证 `1+scale` / `1+gate` 的初始化
   正确、既有 ckpt 可无损续训。
2. **adanorm 模式的旋转不变 / 等变**：`scale` / `gate` 按 `expand_index` 广播是等变性
   最易写错处，单独验证
3. 三种 `scd_inject` 模式 × (direct, 保守力) 全跑通
4. **四档冻结的 `requires_grad=False` 参数计数精确匹配** §3 表中数值
5. 冻结后仍 DDP unused-param 安全
6. 三条编译路径 × adanorm：`cond` 作为 traced 入参、无 stale-bake、编译-eager 数值一致
7. `scd_reg_noise_std > 0` 时 clean 前向确实被扰动
8. **gradient checkpointing 开启时 `cond` 正确透传** —— `_forward_blocks` 中
   checkpoint 分支与普通分支是两段独立代码，加形参时极易只改一处

判定标准沿用 v0：旋转不变 / 等变 `< 1e-4`（实测量级 1e-7），编译-eager 一致 `< 1e-4`。

## 6. 验证配置

新建 `experimental/configs/omat24/mptrj/experiments/direct/scd_v1/eager_fp32_N2L2C64.yml`，
以 `experimental/configs/omat24/mptrj/experiments/direct/compile_test/compile_fp32.yml`
为模板。关键差异：

```yaml
model:
  name:                          equiformer_v3_scd
  scd_inject:                    adanorm
  scd_adanorm_targets:           ['attn', 'ffn']
  scd_adanorm_scope:             per_degree
  scd_p_dropcond:                0.2
  scd_cond_clip:                 100.0
  scd_reg_noise_std:             0.0
  scd_freeze_element_embedding:  none
  scd_freeze_mask_token:         false
  use_force_cond:                true
  drop_path_rate:                0.1      # 模板为 0.05
  # 不设 enable_compile
optim:
  use_compile:                   False    # 模板为 True，此处必须关
```

其余（N2L2C64 结构、HybridMuon/moonlight 参数、DeNS 噪声参数、loss 权重、数据集、
normalizer、element_references）全部照抄模板。不开 amp、不开 tf32、不开任何 compile
→ 纯 fp32 eager。先验证正确性，性能留待后续。

## 7. 交付物

1. `experimental/models/equiformer_v3/layer_norm.py` —— 新增 `EquivariantAdaNorm`
2. `experimental/models/equiformer_v3/transformer_block.py` —— `TransBlockV3` 支持 `cond`
3. `experimental/models/equiformer_v3/equiformer_v3.py` —— `_forward_blocks` / `core_compute` 透传 `cond`
4. `experimental/models/equiformer_v3/equiformer_v3_dens.py` —— DeNS 版 `core_compute` 与
   `_conservative_compiled_forward` 透传 `cond`
5. `experimental/models/equiformer_v3/equiformer_v3_scd.py` —— `scd_inject` 开关、AdaNorm
   构建与 `cond` 下发、四档冻结、`scd_reg_noise_std`
6. `experimental/trainers/equiformer_v3_dens_trainer.py` —— 元素嵌入范数诊断埋点
7. `experimental/tests/test_equiformer_v3_scd.py` —— §5 的 8 项新增断言
8. `experimental/configs/omat24/mptrj/experiments/direct/scd_v1/eager_fp32_N2L2C64.yml`
9. `docs/SCD_V0_NOTES.md` 更新为覆盖 v0 与 v1

## 8. 风险

| 风险 | 缓解 |
|---|---|
| 等变性被 `scale`/`gate` 广播破坏 | §5.2 单独断言；`expand_index` 复用现有已验证的 buffer 模式 |
| `cond` 形参在 gradient checkpointing 分支漏传 | §5.8 专项断言 |
| 编译路径 stale-bake | `cond` 作为显式 traced 入参（方案 A），§5.6 双 batch 断言 |
| DropPath 0.1 在 per-graph 语义下过强 | 列为第一待调项，配置可改；先在 N2L2C64 上观察 |
| 从 DeNS ckpt 续训时 AdaNorm 破坏已学表征 | identity-init（§2.4），§5.1 逐位一致断言 |
