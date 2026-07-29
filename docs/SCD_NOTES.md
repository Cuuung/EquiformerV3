# EquiformerV3-SCD：自条件去噪

对应论文：Perez & Gómez-Bombarelli, *Self-Conditioned Denoising for Atomistic
Representation Learning*（2026）。参考实现：`repositories/scd`（TorchMD-Net 主干）。

## 1. SCD 与 DeNS 的关系

两者是同构的"条件化去噪"，差别只在**条件信号的来源**和**注入位置**：

| | DeNS | SCD（论文） | SCD v0（本实现） |
|---|---|---|---|
| 条件信号 | 真实 DFT 力的球谐编码 | 模型自身对**干净结构**的 L=0 池化嵌入 | 同论文 |
| 注入位置 | 输入层加法 | 每个 block 的 pre-attention AdaNorm | **输入层加法**（同 DeNS） |
| 前向次数 | 1 | 2（clean → 条件 → noisy） | 2 |
| 噪声预测头 | `dens_block` | L=1 等变头 | 复用 `dens_block` |

v0 有意放弃 AdaNorm，换取"零侵入"：不改 `TransBlockV3` / `core_compute` 的签名，
完整复用 DeNS 的 direct / gradient / make_fx-compiled 三条前向路径。AdaNorm 版见 §6。

## 2. 实现要点

改动共 3 处：

- **`experimental/models/equiformer_v3/equiformer_v3_scd.py`**（新增）
  `EquiformerV3SCD_OC(EquiformerV3DeNS_OC)`，注册名 `equiformer_v3_scd`。
- **`experimental/trainers/equiformer_v3_dens_trainer.py`**（+3 行）
  `add_gaussian_noise_to_position` 加噪前存 `batch.pos_clean`。
  （`all_atoms == False` 时只有自由原子被位移，无法由 `pos - noise_vec` 反推。）
- **配置**：`experimental/configs/omat24/mptrj/experiments/direct/equiformer_v3_scd_*.yml`

### 为什么只重写一个方法

DeNS 的三条前向路径都走同一个入口拿输入条件：

```python
force_embedding, noise_mask, dens_batch_mask, dens_mask = self._forward_dens_force_encoding(data)
...
x_scalar, x, ... = core_compute(..., force_embedding)   # core_compute 内部 x = x + force_embedding
```

所以 SCD 只需重写 `_forward_dens_force_encoding`，把自条件项叠加到
`force_embedding` 上即可，**不需要复制任何 forward 代码**。
在编译路径下 `fe` 本就是 traced region 的显式入参（非闭包捕获），
因此既不会 stale-bake，梯度也能穿过编译区回流到 clean 前向。

### 数据流

```
pos_clean --建图--> core_compute --> L=0 特征 --sum-pool--> MLP --> c [B, C]
c --按图 dropout(p=0.2) 替换为 mask token--> LayerNorm --> clamp
  --零初始化 Linear--> 写入 L=0 通道 --> 加到 noisy 前向的输入嵌入
```

- **条件是不变量**：`c` 来自 L=0 通道，只写回 L=0 通道，旋转平移不变性成立
  （测试实测能量 `max|dE| = 1.2e-7`、力 `max|dF| = 5.2e-8`）。
- **零初始化**：`scd_cond_proj` 权重/偏置初始为 0，初始状态与父类 DeNS 逐位一致，
  可直接从既有 DeNS ckpt 续训。
- **单前向退化**：只有"训练态 + 该 step 施加了 DeNS 噪声"才走双前向；
  普通监督 step、验证、推理都是单次建图 + 单次前向，无部署开销。
- **mask token 恒在图中**：`c * keep + mask_token * (1 - keep)`，即使没有图被丢弃，
  `mask_token` 也参与自动微分（梯度为 0 而非 None），DDP
  `find_unused_parameters=False` 安全。

## 3. 配置项

```yaml
model:
  name:             equiformer_v3_scd
  use_scd:          True
  use_force_cond:   True    # False = 论文形态的纯自条件（移除 force_embedding）
  scd_p_dropcond:   0.2
  scd_cond_clip:    100.0
  scd_detach_cond:  False   # True 省一次 backward，但与论文不一致
```

`use_force_cond: False` 会把 `self.force_embedding` 置 None（否则该子模块参数
在 DDP 下变成 unused param）。因此从 DeNS ckpt 加载时需要 `strict=False`。

trainer 仍用 `equiformer_v3_dens_trainer`，噪声注入 / 去噪目标 normalizer /
hybrid loss 全部沿用，不需要新 trainer。

## 4. 开销

去噪 step 的额外成本 = 一次建图 + 一次带梯度的 backbone 前向：

- direct 模式：≈ 2× step 时间（clean 与 noisy 两次完整前向 + 反传）
- 保守力模式：clean 前向不需要 pos 梯度、不建双反向图，开销显著低于 noisy 前向，
  实测量级约 +35–50%

按配置里的 `denoising_pos_params.prob` 摊薄（例如 `prob: 0.5` 时整体开销约为上述一半）。
`scd_detach_cond: True` 可切断 clean 前向的反传再省一部分，但论文让梯度回流，
默认保持 `False`。

## 5. v0 未做的（留给 v1）

- **AdaNorm 注入**：论文把条件经 DiT 式 AdaNorm 送进每层 pre-attention LayerNorm。
  移植到等变主干需要注意 scale/gate 可按通道乘所有 degree，但 **shift 只能加在 L=0**，
  不能照抄 `scd/models/modules/conditioning.py:adaLN2`。
  需要改 `layer_norm.py`（新增 AdaNorm 类）、`transformer_block.py:715`
  （`TransBlockV3.forward` 增加 `cond` 形参）、`equiformer_v3.py` 的
  `_forward_blocks` / `core_compute` 透传，约 150 行新增 + 35 行改动。
- **无标签 SSL 预训练**：v0 跑在有 E/F/S 标签的数据上（SCD 作为附加目标）。
  真正的无标签预训练还需要数据侧支持。

## 6. v1：AdaNorm 条件注入

v1 实现论文形态：把自条件向量经 **AdaNorm** 注入每个 transformer block 的
pre-norm，而不再只用 v0 的输入层加法。沿用同一个模型类
`EquiformerV3SCD_OC`（注册名不变，仍是 `equiformer_v3_scd`），新增
`scd_inject` 开关在两种注入方式间切换 / 叠加。

### 6.1 `scd_inject`：三种注入模式

```yaml
model:
  scd_inject: input | adanorm | both   # 默认 adanorm
```

条件向量的生产链路（clean 前向 → sum-pool → `SCDCondHead` → dropcond →
LayerNorm → clamp）与 v0 完全一致，`scd_inject` 只决定这份向量 `c` 怎么用：

| 值 | 行为 |
|---|---|
| `input` | v0 路径：写入 L=0，加到 `force_embedding`（即噪声前向的输入嵌入） |
| `adanorm` | v1 路径（默认）：`c[data.batch]` 作为 `cond` 透传给每个 `TransBlockV3`，调制其 pre-attention / pre-FFN norm |
| `both` | 两者叠加，但只算一次 `c`（不会跑两次 clean 前向） |

`scd_inject` 只控制 **SCD 自条件** 这一条路径；DeNS 的真实力条件仍由独立的
`use_force_cond` 控制，两者正交
（见 `experimental/models/equiformer_v3/equiformer_v3_scd.py:_forward_dens_force_encoding`
与 `_forward_cond`）。

`adanorm` / `both` 模式下，构造函数会调用 `_rebuild_blocks_with_adanorm`
按 `scd_adanorm_targets`（默认 `('attn', 'ffn')`，可传 `('attn',)` 做消融）
重建 `self.blocks`，把对应的 `norm_1` / `norm_2` 换成 `EquivariantAdaNorm`；
未列入 targets 的位置仍是普通 `get_normalization_layer`。

### 6.2 `EquivariantAdaNorm`：等变性论证与 identity-init

新模块：`experimental/models/equiformer_v3/layer_norm.py::EquivariantAdaNorm`。
包住现有的等变 norm，由条件向量 `cond` 额外产出 `(shift, scale, gate)`：

```python
x = self.norm(x)                              # 复用现有等变 norm
inp = cat([cond, x[:, 0, :].detach()], -1)     # use_node_feat=True 时拼节点自身 L=0
shift, scale, gate = self.fc(inp).split(...)   # fc 末层零初始化
x = x * (1 + scale[:, expand_index, :])
x = x + shift_padded_to_L0                     # out-of-place，只写 L=0
return x, 1 + gate[:, expand_index, :]
```

**等变性论证**：`self.norm` 本身等变（沿用现有实现）；`scale` / `gate` 是不变
标量（源自 `cond` 与 `x` 的 L=0 分量，二者都是旋转不变量），且经
`expand_index`（复用 `EquivariantMergeLayerNorm` 已有的 buffer 模式）广播后，
**同一 `(l, c)` 的全部 `2l+1` 个 m 分量共享同一个标量**——这正是等变性的充要
条件，因此乘上去仍等变；`shift` 只加在 L=0（用 `l0_mask` 构造一个只有 L=0
非零的张量再相加，避免 in-place 写入破坏 autograd version-counter /
`make_fx`），故加上去仍等变。测试实测（N=7 节点、随机旋转）
`max|d| ≈ 9.54e-07`（`per_degree`/`shared` scope）～`4.77e-07`（`l0_only`
scope），远低于判定阈值 `1e-4`。

**identity-init（不是 DiT 的 zero-init gate）**：`fc` 最后一层 `weight` /
`bias` 全零初始化，但读出时用 `1 + scale`、`1 + gate`（`shift` 直接加 0）。
DiT 的 adaLN-Zero 让 `gate = 0`，残差支在 step 0 完全关闭——那是从零训练的
设定；本仓库的 AdaNorm 需要能从既有 equiv3 / DeNS ckpt 续训，`gate = 0` 会
把预训练 backbone 整个关掉。用 `1 + gate` 后，**未训练时 `EquivariantAdaNorm`
逐位等价于原 norm**，`TransBlockV3` 在有无 `cond` 两种调用下输出逐位相同
（`experimental/tests/test_equiformer_v3_adanorm.py` 中
`identity-init: 输出等于原 norm max|d|=0.00e+00`、
`block identity-init: 有无 cond 输出一致 max|d|=0.00e+00`）。

`scd_adanorm_scope` 三档（`experimental/models/equiformer_v3/layer_norm.py`）：

| scope | `gate` 形状 | 语义 |
|---|---|---|
| `per_degree`（默认） | `[N, (lmax+1)**2, C]` | 每个 degree 独立的 scale/gate |
| `shared` | `[N, 1, C]`（不展开，靠广播） | 全 degree 共享同一 scale/gate |
| `l0_only` | `[N, (lmax+1)**2, C]`，但 L≥1 恒为 1 | 只调制 L=0，对齐参考实现 `dx*gate_x, dvec` 的严格形态 |

`scd_adanorm_use_node_feat`（默认 `True`）控制条件 MLP 的输入是否额外拼上
该节点自身的 L=0 特征（`detach()` 后拼接，不参与该分支的反传）。

### 6.3 元素嵌入冻结（四档）

`scd_freeze_element_embedding: none | sphere | sphere_edge | all`
（`experimental/models/equiformer_v3/equiformer_v3_scd.py::_apply_element_embedding_freeze`）。
equiv3 的元素身份有三个入口共 `1 + 2 + 2×num_layers` 张 embedding 表
（`sphere_embedding` 1 张、`EdgeDegreeEmbedding.{source,target}_embedding` 2 张、
每个 attention block 的 `ga.{source,target}_embedding` 2 张 × 层数），N@7
（`num_layers=7`）为 17 张、N2L2C64（`num_layers=2`）为 7 张，比论文参考实现
（ET，仅 1 张）多得多，因此拆成四档递进冻结。**输出头（`force_block` / `dens_block` /
`stress_block`）明确不在冻结范围**——它们是任务头而非输入通道。

以下 `requires_grad=False` 参数量在本任务的配置里已实测核对（脚本对每档
分别实例化模型、直接统计冻结参数总数）：

| 档位 | 冻结对象 | N2L2C64（本配置） | N@7L@4C@128 |
|---|---|---|---|
| `none`（默认） | — | 0 | 0 |
| `sphere` | `sphere_embedding` | 8,192 | 16,384 |
| `sphere_edge` | ↑ + `edge_degree_embedding.{source,target}_embedding` | 40,960 | 49,152 |
| `all` | ↑ + 每个 `blocks[i].ga.{source,target}_embedding` | 106,496 | 278,528 |

`scd_freeze_mask_token`（默认 `False`）独立控制是否额外冻结
`scd_mask_token`。冻结参数不进 autograd 图，HybridMuon（`muon.py`）与
`base_trainer.py` 都会跳过 `requires_grad=False` 的参数，DDP
`find_unused_parameters` 安全。

**本配置默认 `none`**：只冻 `sphere_embedding` 未必能阻止塌缩——元素信息仍
可能从另外两个入口进入。配套诊断埋点见 §6.5，先用 `none` 档跑一次训练看
三组范数的实际走向，再决定要不要冻、冻到哪档。

### 6.4 正则化默认值

| 项 | 论文（Table 17 / 附录 A.1） | v0 现状 | v1 默认 | 接口 |
|---|---|---|---|---|
| 条件 dropout | 0.2 | 0.2（v0 已有） | 0.2 | `scd_p_dropcond` |
| DropPath | 0.1 | 0.05 | **0.1** | `drop_path_rate` |
| clean 前向正则噪声 σ | 0.005 | 无 | **0.0**（新增开关，默认关） | `scd_reg_noise_std` |
| 腐蚀噪声 σ | 0.04 | 0.025 | 0.025 不动 | `denoising_pos_params.std` |
| 条件数值截断 | ±100 | ±100（v0 已有） | ±100 | `scd_cond_clip` |
| 元素嵌入冻结 | 冻（单表） | 无 | `none` | `scd_freeze_element_embedding` |

**DropPath 0.05 → 0.1**：论文把这条系在双前向训练上（附录 B：双前向下
`drop path 0.1` 提升稳定性）。v1 的 `scd_inject=adanorm` 正是双前向。**注意**
equiv3 的 `GraphDropPath`（`drop.py`）是 **per-graph** 整图丢弃，参考实现的
`JointDropPath` 是 **per-node**——同一数值下本仓库的正则强度更大，
是本配置**第一个待调项**，不是照搬论文数值就完事。

**`scd_reg_noise_std` 默认 0**：参考实现只在 `noise_in_loader=False` 分支
施加该正则噪声，而其材料预训练配置用的是 `noise_in_loader=True`——作者自己
在材料数据上就没走这条分支，本项目无正面证据支持默认开启，因此保留开关但
默认为 0（见 `_scd_clean_cond` 中的实现与注释）。

### 6.5 验证：三个测试文件

```bash
python experimental/tests/test_equiformer_v3_adanorm.py     # EquivariantAdaNorm 单元测试
python experimental/tests/test_element_embedding_diag.py    # 元素嵌入范数诊断埋点
python experimental/tests/test_equiformer_v3_scd.py         # 端到端回归（含 v0 + v1）
python experimental/tests/test_scd_v1_config.py             # v1 配置解析 + 模型实例化，很快
```

- **`test_equiformer_v3_adanorm.py`**：`EquivariantAdaNorm` 与 `TransBlockV3`
  的单元测试，覆盖 identity-init 逐位一致、三种 scope 下的旋转等变 / gate
  旋转不变、`use_node_feat` 开关、`adanorm_targets` 部分启用时另一路退回普通
  norm 等。
- **`test_element_embedding_diag.py`**：验证 §6.3 提到的诊断函数
  `experimental/trainers/equiformer_v3_dens_trainer.py::element_embedding_norms`——
  返回三组均值范数、置零某表后对应范数归零且不影响其余两组、对
  `equiformer_v3_dens`（无 SCD）同样可用、置零输出头（`force_block`）不影响
  `emb_norm_blocks`（证明该函数确实只统计输入侧的元素嵌入）。
- **`test_equiformer_v3_scd.py`**：在 v0 的注册 / 零初始化等价性 / 条件只占
  L=0 / direct-保守力两条前向 / 梯度连通性 / 编译一致性等断言基础上，扩展了
  AdaNorm 相关断言（identity-init 逐位一致、adanorm 模式旋转不变/等变、三种
  `scd_inject` × direct/保守力全跑通、四档冻结参数计数精确匹配 §6.3 表中数值、
  冻结后 DDP unused-param 安全、三条编译路径下 `cond` 正确透传且编译-eager
  数值一致、`scd_reg_noise_std > 0` 时 clean 前向确实被扰动、gradient
  checkpointing 开启时 `cond` 正确透传）。该文件含 4 段
  `torch.compile`，运行约 14 分钟，需 GPU 与项目训练镜像。
- **`test_scd_v1_config.py`**：只做配置解析 + 模型实例化，不跑前向，验证
  `experimental/configs/omat24/mptrj/experiments/direct/scd_v1/eager_fp32_N2L2C64.yml`
  的关键开关（`scd_inject=adanorm`、blocks 全部建成 AdaNorm、
  `optim.use_compile=False`、`drop_path_rate=0.1`、冻结档默认 `none`、
  `scd_reg_noise_std` 默认 0、N2L2C64 结构、trainer 名）。运行很快。

### 6.6 v1 验证配置

`experimental/configs/omat24/mptrj/experiments/direct/scd_v1/eager_fp32_N2L2C64.yml`
以 `experimental/configs/omat24/mptrj/experiments/direct/compile_test/compile_fp32.yml`
为模板，仅改三处：插入 §6 的 SCD/AdaNorm 开关块、`drop_path_rate: 0.05 → 0.1`、
`optim.use_compile: True → False`。不设 `enable_compile`（那是另一条互斥的
编译路径），不开 amp、不开 tf32——纯 fp32 eager，v1 的交付目标是先验证
正确性，性能与编译留待后续。其余（数据集、normalizer、element_references、
loss 权重、outputs、N2L2C64 结构、HybridMuon/moonlight 参数、DeNS 噪声参数）
逐字沿用模板。

## 7. 参考

设计文档：`docs/superpowers/specs/2026-07-29-scd-v1-design.md`（架构决策、
已否决方案、风险表）。
