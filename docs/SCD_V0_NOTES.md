# EquiformerV3-SCD v0：自条件去噪插件

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
完整复用 DeNS 的 direct / gradient / make_fx-compiled 三条前向路径。AdaNorm 版见 §5。

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

## 6. 验证

```bash
python experimental/tests/test_equiformer_v3_scd.py
```

覆盖 20 项：注册、零初始化等价性、条件只占 L=0、direct/保守力两条前向、
梯度连通性与 DDP unused-param 安全、单/双建图次数、eval 单前向、
旋转不变/等变、`use_force_cond=False` 分支、`no_weight_decay`、
外层 `torch.compile` 与内层 `make_fx` 两条编译路径、编译-eager 数值一致性
（`max|dE| = 1.2e-7`）。需 GPU 与项目训练镜像。
