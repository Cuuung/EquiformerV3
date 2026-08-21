# EquiformerV3 模型架构详解（数据结构 + 算子类型）

> 本文严格对照 `experimental/models/equiformer_v3/` 下的源码逐环节梳理。
> 所有形状、算子、精度岛均标注了对应的 `文件:行号`，可直接核对。
> 具体维度以当前 gradient finetune 配置为实例：
> **`num_layers N=7`, `num_channels C=128`, `lmax L=4`, `mmax M=2`,
> `num_heads=8`, `attn_alpha=64`, `attn_value=16`, `attn_hidden=32`,
> `ffn_hidden=512`, `num_radial_basis=10`, `max_radius=6.0`, `max_neighbors=20`,
> `norm_type=merge_layer_norm`, `activation=sep-merge_gates2_swiglu`,
> `use_add_merge=False`, `use_rad_l_parametrization=True`。**

---

## 0. 核心数据结构：等变 irreps 张量

整个网络里流动的"节点/边特征"不是普通的 `[N, C]` 向量，而是**带球谐（spherical harmonic）结构的等变张量**：

```
节点特征 x     :  [N_nodes, (L+1)², C]   →  L=4 时  [N,  25, 128]
边消息   x_msg :  [N_edges, (L+1)², C]   →         [E,  25, 128]
```

- 第 1 维 `N`：节点数（或边数 `E`）。
- 第 2 维 `(L+1)² = 25`：**球谐分量**，按 degree `l=0..4` 分块排布，第 `l` 块长度 `2l+1`（1,3,5,7,9），下标区间 `[l², l²+2l+1)`。
  - `l=0`（下标 0）是**标量**（旋转不变），能量、gating、attention 权重都从这里取。
  - `l≥1` 是**等变**分量，随坐标系旋转而按 Wigner-D 变换；`l=1`（下标 1,2,3）对应向量，力就从这里读出。
- 第 3 维 `C=128`：通道数。

> 这是等变模型和普通 GNN 最根本的差别：每个特征都被 `(L+1)²=25` 撑大了 25 倍。
> 这也是访存/算力开销的主要来源之一。

**mmax 截断（SO(2) 卷积技巧）**：`mmax=2 < lmax=4`。把边特征旋转到"边方向对齐 +Y 轴"的规范系后，只需保留 `|m| ≤ mmax` 的分量，`(L+1)²=25` 个分量被压到 **19** 个（`l=0:1, l=1:3, l=2:5, l=3:5, l=4:5`）。等变卷积因此退化成便宜的 SO(2) 线性，算完再旋转回 25 维。（`so3.py:317-338`、`so2_ops.py:111-151`）

---

## 1. 整体前向流程总览

`forward` 按 `direct_prediction` 分两条路径（`equiformer_v3.py:861-866`）：

- **direct**（直接预测力）：`_forward_direct`（`equiformer_v3.py:550-618`）
- **gradient**（保守力，本 finetune 用）：`_forward_gradient`（`equiformer_v3.py:621-724`），能量对坐标求导得力，训练时是双反向。

两条路共享核心计算体 `core_compute`（`equiformer_v3.py:517-547`）：

```
data(pos, atomic_numbers, cell, batch)
        │
        ▼  ① generate_graph                    [eager, dynamo-disable]
edge_index[2,E] · edge_distance[E] · edge_distance_vec[E,3]
        │
        ▼  ② _forward_edge                      [几何路径, fp32]
Wigner-D wigner/wigner_inv[E,19,25] · rbf[E,10] · envelope[E,1]
        │
        ▼  ③ _forward_embedding                 [嵌入]
x [N, 25, 128]
        │
        ▼  ④ _forward_blocks   × 7              [Transformer 主体]
x [N, 25, 128]
        │
        ▼  ⑤ final norm → x_scalar[N,128]
        │
        ▼  ⑥ 输出头：energy / force / stress
outputs
```

> 两个子类 `EquiformerV3DeNS_OC` / `EquiformerV3SCD_OC` 都挂在这条主干上：
> 它们不改 ①②④⑤，只在 ③ 之后往 `x` 上叠加条件、在 ⑥ 处并联一个去噪头，
> SCD 另外把 ④ 里的两处 pre-norm 换成 AdaNorm。详见 **§6**。

---

## 2. 逐环节详解

### ① 建图 generate_graph（`equiformer_v3.py:562 / base.py:70-155`）

从原子坐标现算邻居图（on-the-fly，`otf_graph=True`）。周期性体系走 `radius_graph_pbc`（`common/utils.py:618-759`）。

| 输出 | 形状 | 含义 |
|---|---|---|
| `edge_index` | `[2, E]` | 每条边的 (source, target) 节点下标 |
| `edge_distance` | `[E]` | 边长（标量） |
| `edge_distance_vec` | `[E, 3]` | 边方向向量（source→target） |
| `cell_offsets` | `[E, 3]` | 周期镜像偏移 |

- **算子类型**：`repeat_interleave` / `masked_select` / `nonzero` / `cdist`（`common/utils.py:592,652,753`）。**数据依赖形状**（边数未知），且 `generate_graph` 带 `@torch._dynamo.disable`（`equiformer_v3.py:926`），故这段在 eager 下运行、不进 `torch.compile`。
- **在梯度力编译路径里**：建图只提供连通性（不可导），真正的边几何 `edv/ed` 在编译区内用 `index_select` + `norm` 从 `pos` 重算，保证力精确（`equiformer_v3.py:780-785`）。

### ② 边几何路径 _forward_edge（`equiformer_v3.py:401-418`）— fp32

把边方向变成旋转算子和径向特征。**这是等变性的数值命脉，全程 fp32**（bf16 会破坏等变、fp16 会崩 Wigner）。

**2a. 欧拉角** `init_edge_rot_euler_angles`（`edge_rot_mat.py:52-67`）
```
edge_distance_vec[E,3] ──normalize+clamp──> 单位方向
  beta  = Safeacos(y)          # 离 +Y 的极角
  alpha = Safeatan2(x, z)      # XZ 平面方位角
  gamma = rand()*2π            # 训练随机滚转 (SO(2) 等变)
→ eulers = (-gamma, -beta, -alpha)   每个 [E]
```
- **算子**：`F.normalize`、`clamp`、`Safeacos`/`Safeatan2`（自定义 autograd.Function，带 clamp-safe 反向，防 `|x|→1` 处 NaN，`edge_rot_mat.py:21-49`）、`torch.rand`。全部逐元素。

**2b. Wigner-D 矩阵** `set_wigner_from_eulers`（`so3.py:357-383`）→ `eulers_to_wigner`（`edge_rot_mat.py:70-98`）
```
每个 l:  wigner_D(l,α,β,γ) = Xa @ J_l @ Xb @ J_l @ Xc     # wigner.py:16-27
         (Xa/Xb/Xc 是 z 轴旋转矩阵, sin/cos 填入, wigner.py:30-38)
块对角拼装 → [E, 25, 25]
einsum('mi,nij->nmj', wigner_index_to_m_array, ·) → 截断到 m≤mmax → wigner [E, 19, 25]
transpose + rescale → wigner_inv [E, 25, 19]
```
- **算子**：每个 `l` 的 `Xa@J@Xb@J@Xc` 是 4 次**小矩阵乘**（`(2l+1)×(2l+1)`，最大 9×9）；`_z_rot_mat` 用 `index_put`(sin/cos) 填充；`F.pad` 拼块对角；`einsum` + `transpose`。梯度模式下保留梯度、direct 模式下 `detach`（`so3.py:379-381`）。

**2c. 包络 + 径向基**
```
envelope_func(edge_distance)   多项式平滑截断     → [E, 1]   (envelope.py:20-30, torch.where)
distance_expansion(edge_distance)  GaussianSmearing → [E, 10]  (radial_function.py:18-20, exp)
```

### ③ 嵌入 _forward_embedding（`equiformer_v3.py:421-454`）

初始化节点 irreps 特征 `x [N, 25, 128]`（全零），再填两部分：

**3a. 原子类型嵌入**（`equiformer_v3.py:442-443`）
```
sphere_embedding(atomic_numbers)  nn.Embedding → [N, 128]
放到标量位:  x[:, 0, :] = atom_embedding
```
- **算子**：embedding lookup（gather）。

**3b. 边度嵌入** `EdgeDegreeEmbedding`（`input_block.py:74-112`）
```
source/target embedding + rbf ──cat──> x_edge [E, 10+2·128]
rad_func(x_edge)   Linear+LN+SiLU → [E, 5·128] ─×envelope─ reshape [E, 5, 128]
bmm(wigner_inv[:,:,:5], x_edge_m0) → [E, 25, 128]   # 把 m0 特征旋转成完整 irreps
reduce_edge (scatter index_add)   → [N, 25, 128]  / avg_degree
```
- **算子**：`nn.Embedding`、`nn.Linear`/`LayerNorm`/`SiLU`（radial）、**`torch.bmm`**（Wigner 逆旋转，`input_block.py:96`）、**`index_add_` scatter**（`utils.py:13`，fp32 累加）。
```
x = atom_embedding + edge_degree_embedding    → [N, 25, 128]
```

### ④ Transformer 主体 _forward_blocks（`equiformer_v3.py:457-514`）× 7

每层 `TransBlockV3`（`transformer_block.py:715-759`）是**双残差 pre-norm** 结构：

```
x → norm_1 → EquivariantGraphAttention → +x
  → norm_2 → FeedForwardNetwork        → +x
```

> bf16 混合精度时，`use_amp` 的 autocast 只包住这 7 层块（`equiformer_v3.py:475-509`）；块前的几何/嵌入、块内的 norm、块后的输出头都在 fp32。

#### ④-A 等变图注意力 EquivariantGraphAttention（`transformer_block.py:246-350`）

这是全模型最重的模块，是"非线性消息传递 + MLP attention"。数据流：

```
1. 边标量特征                                                      transformer_block.py:259-269
   source/target embedding + rbf ──cat──> x_edge
   rad_func(x_edge)  → x_edge_weight [E, 25, 2·128]        (Linear+LN+SiLU, radial)

2. 展开源/目标节点特征到边  (fp32 岛)                                transformer_block.py:276-281
   x_f = x.float()
   x_source = index_select(x_f, edge_index[0])   → [E, 25, 128]   (gather)
   x_target = index_select(x_f, edge_index[1])   → [E, 25, 128]   (gather)
   x_message = cat(x_source, x_target)           → [E, 25, 256]

3. 径向加权 + 旋转到局部系                                          transformer_block.py:282-284
   x_message = x_message * x_edge_weight          (elementwise)
   x_message = so3_rotation.rotate(x_message)     bmm(wigner, ·) → [E, 19, 256]

4. SO(2) 线性 1  (含 attention/gating 的 extra_m0)                 transformer_block.py:297
   so2_linear_1 → x_message[E,19,H], x_m0_extra[E, extra]
       m=0: fc_m0 (nn.Linear);  m>0: SO2MLinear.fc (nn.Linear)   so2_ops.py:46-60,111-151

5. S2 grid 激活 (sep-merge_gates2_swiglu)                          transformer_block.py:300-311
   split x_m0_extra → x_alpha[E,heads·alpha], x_scalar (gating)
   act: to_grid(einsum) → gated-SwiGLU MLP → from_grid(einsum)

6. SO(2) 线性 2                                                     transformer_block.py:313
   so2_linear_2 → value [E, 19, heads·value=128]

7. 注意力权重  (标量路)                                             transformer_block.py:316-333
   x_alpha[E,heads,64] → alpha_norm(LayerNorm) → SmoothLeakyReLU
   alpha = einsum('bik,ik->bi', x_alpha, alpha_dot)   → [E, heads]
   alpha = GraphSoftmax(alpha, edge_index[1])          scatter max/sum 归一
   attn = value * alpha                                → [E, 19, 128]

8. 旋转回全局系 + 聚合到节点                                        transformer_block.py:338-345
   rotate_inv(attn)   bmm(wigner_inv, ·) → [E, 25, 128]
   reduce_edge (scatter index_add)       → [N, 25, 128]

9. 投影                                                            transformer_block.py:348
   proj = SO3Linear → [N, 25, 128]
```

**算子类型拆解**：

| 步骤 | 算子 | 类型 |
|---|---|---|
| 1 | `nn.Embedding` + `nn.Linear`/`LayerNorm`/`SiLU` | lookup + 稠密 GEMM + norm |
| 2 | `index_select` ×2 | **gather（访存）**，fp32 岛 |
| 3 | `mul` + `bmm(wigner,·)` | elementwise + **batched 小矩阵乘** |
| 4/6 | `SO2Linear`（`nn.Linear`） | **稠密 GEMM（TensorCore）** |
| 5 | `to_grid`/`from_grid`（einsum）+ 网格 MLP（Linear/sigmoid/mul） | GEMM + 逐元素 gate |
| 7 | `LayerNorm` + `einsum` + `GraphSoftmax`（scatter max/sum） | norm + **scatter 归约** |
| 8 | `bmm(wigner_inv,·)` + `index_add_` | **batched 小矩阵乘** + **scatter（访存）**，fp32 |
| 9 | `SO3Linear` | **稠密 GEMM（TensorCore）** |

#### ④-B 前馈网络 FeedForwardNetwork（`transformer_block.py:353-541`）

`use_grid_mlp=True` + `sep-merge_gates2_swiglu`：

```
inputs[N,25,128]
  标量路: scalar_mlp(inputs[:,0:1])          LinearSwiGLU        transformer_block.py:489-490
  so3_linear_1(inputs) → [N,25, 2·512]        SO3Linear (GEMM)    transformer_block.py:496
  to_grid → [N, grid_pts, H]                  einsum              transformer_block.py:500
  GatedSwiGLUGridMLP(grid, inputs[:,0:1])     Linear×3+sigmoid+mul transformer_block.py:544-567
  from_grid → [N,25,H]                        einsum              transformer_block.py:518
  合并标量路 (l=0 相加)                                            transformer_block.py:527-529
  so3_linear_2 → [N, 25, 128]                 SO3Linear (GEMM)    transformer_block.py:539
```

- **算子类型**：`SO3Linear`（GEMM）、`to_grid/from_grid`（einsum GEMM）、grid MLP（`nn.Linear` + `Sigmoid` + `mul`）、`LayerNorm`。

#### ④-C 归一化 EquivariantMergeLayerNorm（`layer_norm.py:194-278`）— fp32

```
标量位去均值 (centering) → 全 degree 求 pow(2).mean(通道) → 按 degree 平衡加权 (einsum)
→ rsqrt → affine (index_select 展开权重) → 乘回
```
- **算子**：`mean`/`pow`/`rsqrt` 归约 + `einsum` + `index_select`。
- **精度**：类装饰器 `@torch.cuda.amp.autocast(enabled=False)`（`layer_norm.py:246`）强制 fp32，**无论外层开不开 amp 都生效**。

### ⑤⑥ 输出头

**最终归一化 + 取标量**（`equiformer_v3.py:511-514`）
```
x = self.norm(x)                       EquivariantMergeLayerNorm (fp32)
x_scalar = x[:, 0, :] → [N, 128]
```

**能量**（`equiformer_v3.py:589-593 / 693-697`）
```
energy_block(x_scalar)   ScalarFFN: Linear-SiLU-Linear → node_energy[N,1]   output_block.py:64-69
energy = zeros[B].index_add_(0, batch, node_energy)   → [B]  / avg_num_nodes
```
- **算子**：`nn.Linear`/`SiLU`（稠密 GEMM）+ `index_add_`（scatter）。

**力**：

- **gradient 路径（本 finetune）**（`equiformer_v3.py:715-722`）
  ```
  forces = -autograd.grad(energy.sum(), pos, create_graph=self.training)[0]  → [N, 3]
  ```
  - **算子类型**：`torch.autograd.grad`。训练时 `create_graph=True` → **双反向**（把整个前向再反向一遍构二阶图），显存/算力约为 direct 的 3–4 倍。
- **direct 路径**（`equiformer_v3.py:596-607`）
  ```
  force_block = EquivariantGraphAttention(..., activation='gate', out=1)
  forces = out.narrow(1, 1, 3).view(-1, 3)      # 取 l=1 向量分量
  ```

**应力（可选）**（`output_block.py:168-214 / 264-298`）：等变头 → 取 `l=2` 的 9 分量 → `scatter(mean)` → 乘 CG 变换矩阵 `einsum`。

---

## 3. 全模型算子类型汇总

| 算子类别 | 代表实现 | 出现位置 | 硬件特性 |
|---|---|---|---|
| **稠密矩阵乘（TensorCore）** | `SO3Linear`(einsum `bmi,moi->bmo`)、`SO2Linear`/`nn.Linear`、`to_grid`/`from_grid` einsum、grid/scalar/energy MLP | FFN、attention 的 SO2、投影、能量头 | 计算密集，但受 `(L+1)²` 撑大、`C=128` 偏小、按头/按 l 拆分 → **多为小/瘦 GEMM，张量核利用率低** |
| **batched 小矩阵乘（bmm）** | `so3_rotation.rotate`/`rotate_inv`、`edge_degree` 的 wigner bmm、`wigner_D` 的 `Xa@J@Xb@J@Xc` | 每条边的 Wigner 旋转 | 每边独立小矩阵（≤25×25），大 batch，**等变性固有开销** |
| **gather（访存）** | `index_select`、`nn.Embedding` | 节点特征聚到边、原子/边嵌入 | 访存受限；节点→边 gather 是 fp32 岛（`transformer_block.py:276`） |
| **scatter（访存）** | `reduce_edge`(`index_add_`)、`GraphSoftmax`(scatter)、能量/应力聚合 | 边→节点消息聚合、注意力归一 | 访存受限；`reduce_edge` 强制 fp32 累加（`utils.py:5-13`，bf16 无硬件 atomic-add） |
| **逐元素 / 激活** | `SiLU`/`SwiGLU`/`Sigmoid` gate、`GaussianSmearing` exp、envelope、`mul` | radial、grid MLP、gating | 访存受限 |
| **归约 / 归一化（fp32）** | `EquivariantMergeLayerNorm`、`LayerNorm`、softmax 的 max/sum | 每个 block 两次 norm + 注意力 | fp32 守卫（装饰器强制） |
| **几何（fp32）** | `Safeacos`/`Safeatan2`/`normalize`、`wigner_D` | `_forward_edge` | 等变数值核心，全程 fp32 |
| **自动微分** | `torch.autograd.grad`（`create_graph`） | 保守力 | 训练时双反向 |

---

## 4. 精度岛（fp32 保护点）小结

模型即便在 bf16/amp 下也刻意保留若干 fp32 区域，原因见对应代码注释：

| fp32 区域 | 位置 | 原因 |
|---|---|---|
| 边几何 + Wigner 构造 | `_forward_edge`（几何路径，在块 autocast 区外） | 等变数值核心；fp16 会在 `so3.rotate` 触发 `bmm(half,float)` 失配崩溃 |
| 所有等变 LayerNorm/RMSNorm | `layer_norm.py:53,246,…` 装饰器 | 归约在低精度下数值不稳（AMP 标准做法） |
| 节点→边 gather | `transformer_block.py:276-278` | 让其反向 scatter-add 落在 fp32（bf16 无 atomic-add，反向退化成慢 aten 核 ~12×） |
| 边→节点 scatter | `utils.py:5-13` `reduce_edge` | 同上，保持融合的 triton scatter + 数值更稳 |
| 块输出转回 fp32 | `equiformer_v3.py:508-509` | norm 与能量/力头在 fp32 |

> 注：以上 fp32 岛在 tf32 / 纯 fp32 训练下均为 no-op（`.float()`/`.to()` 不改变 dtype）。

---

## 5. 两条前向路径对照

| | direct（`_forward_direct`） | gradient（`_forward_gradient`，本 finetune） |
|---|---|---|
| 力来源 | 等变注意力头直接回归 `l=1` 分量 | 能量对 `pos` 求导 `-∂E/∂pos` |
| 反向 | 单反向 | 训练时**双反向**（`create_graph=True`） |
| Wigner 梯度 | `detach`（不回传旋转，`so3.py:379-381`） | 保留梯度（`use_rotation_mask=True`） |
| 编译 | `core_compute` 走 `plain_compile`（`equiformer_v3.py:573-577`） | 训练态走 `_conservative_compiled_forward`，整段双反向进编译区（`equiformer_v3.py:727-858`）；eval 走 eager |
| 显存/算力 | 基准 | ≈ 3–4× |

---

## 6. 模型变体：DeNS 与 SCD

前面五节讲的是基类 `EquiformerV3_OC`。仓库里还有两个子类，用**同一套主干**做去噪自监督：

```
EquiformerV3_OC                       equiformer_v3.py:…        基础 E/F/S 回归
   └── EquiformerV3DeNS_OC            equiformer_v3_dens.py:24  + 去噪头 + 真实力条件
          └── EquiformerV3SCD_OC      equiformer_v3_scd.py:55   + 自条件（AdaNorm 注入）
```

注册名分别是 `equiformer_v3` / `equiformer_v3_dens` / `equiformer_v3_scd`，三者共用
`equiformer_v3_dens_trainer`。

### 6.1 DeNS：去噪作为辅助任务

**核心思路**：训练时以概率 `prob` 给坐标加高斯噪声，让模型预测"把原子推回原位"的位移。
噪声位移和力都是逐原子的 `l=1` 向量，所以可以复用同一个输出槽。

#### 6.1-A 训练端：加噪（`equiformer_v3_dens_trainer.py:49-202`）

```
batch.pos_clean = batch.pos.clone()                              trainer:188   (SCD 用)
noise_vec  ~ N(0, std²)                          [N, 3]          trainer:81-82
batch.pos += noise_vec        (all_atoms=False 时只动自由原子)     trainer:190-196
batch.noise_vec           = noise_vec            → 去噪的回归目标
batch.denoising_pos_forward = True               → 模型据此切换到去噪模式  trainer:199
batch.dens_batch_mask                            → 哪些图施加了 DeNS（用于屏蔽 stress）
```

`corrupt_ratio` 非 null 时只腐蚀一部分原子，产生逐原子的 `batch.noise_mask`，
未腐蚀的原子仍拟合真实力——这就是 DeNS 的"混合任务"形态。

#### 6.1-B 模型端新增的两个模块

| 模块 | 定义 | 形状 | 作用 |
|---|---|---|---|
| `force_embedding` | `SO3Linear(1 → C, lmax)`（`dens:230-235`） | `[N,25,1] → [N,25,128]` | **输入侧条件**：把真实力编码进节点嵌入 |
| `dens_block` | `EquivariantGraphAttention(out=1)`（`dens:238-262`） | `[N,25,128] → [N,25,1]` | **输出侧去噪头** |

> `dens_block` 与 `force_block` 是**同一个类**的两个实例，不是轻量读出头——
> 内部是完整的一轮等变图注意力（§2 ④-A 那整套）。区别只有 `num_out_channels=1`
> 和四个 dropout 强制为 0。这是 DeNS 主要的额外开销来源。

**真实力条件的编码**（`_forward_dens_force_encoding`，`dens:742-754`）：

```
force_data[N,3] ──e3nn.spherical_harmonics(normalize=True)──> force_sh[N,25]   dens:732-737
force_norm = |F| / √3                                          [N,1]           dens:745-746
force_embedding = force_sh * force_norm → view[N,25,1]
                → SO3Linear → [N,25,128]                                       dens:749
                → × noise_mask（只有被加噪的原子拿到条件）                        dens:751
```

方向走球谐、模长走标量，两者相乘后过 `SO3Linear` —— 这样条件本身是等变的，可以直接加到 `x` 上：

```
core_compute:  x = _forward_embedding(...) ;  x = x + force_embedding          dens:303
```

这是 DeNS 对 `core_compute` 的**唯一**改动（`dens:280-314`），③ 之后、④ 之前。

#### 6.1-C 输出合流

去噪头和力头的输出在**同一个 `outputs['forces']` 槽**里按原子级掩码二选一
（`dens:392` direct / `dens:527` gradient / `dens:705-708` compiled）：

```python
outputs['forces'] = forces * (~noise_mask_tensor) + denoising_pos_vec * noise_mask_tensor
```

同时 stress 在 DeNS 的图上被屏蔽（结构已被扰动，真实 stress 不再是标签）：

```python
outputs['stress'] = stress * (~dens_batch_mask_tensor)                        dens:401/504/667
```

损失端对应地换靶（`trainer:753-…`）：`denoising_pos_forward` 为真时，
`forces` 的 target 从 `batch.forces` 换成**归一化后的 `batch.noise_vec`**（`trainer:765,779`），
归一化因子是噪声本身的 `std`（`trainer:448-456`）。评测也分流成 `denoising_pos_mae` / `denoising_force_mae`
两个指标（`denoising_pos_eval`，`trainer:205-249`）。

#### 6.1-D 一个结构性后果：零梯度而非无梯度

纯去噪配置（`prob=1.0`、`corrupt_ratio=null`、`all_atoms=True`）下 `noise_mask` 全为 True，
于是 `forces * (~noise_mask)` **恒为 0**：`force_block` 每步都拿到零梯度，但仍留在 autograd 图里。
这是刻意的——DDP 默认 `find_unused_parameters=False`，参数拿不到梯度会直接报 reduction 错误；
乘 0 保证它"参与但不学习"。`energy_block` 走的是同一套逻辑（靠 `coefficient: 0` 的损失项留在图里）。

> 注意零梯度 ≠ 冻结：解耦权重衰减对有梯度张量照常执行（`muon.py:359-360` 只在 `p.grad is None` 时跳过），
> 这两个头会被缓慢缩小。下游微调时它们等价于随机初始化。

#### 6.1-E 编译路径的代价

保守力编译区（`_conservative_compiled_forward`，`dens:532-710`）里，`dens_block` 需要等变特征 `x`，
而编译区只吐能量和力。所以实现上**额外跑了一整次 eager `core_compute`** 专门喂它（`dens:686-695`）。
direct 路径没有这个问题——`_forward_direct` 只算一次 `x`，`force_block` 和 `dens_block` 共用（`dens:369-390`）。

### 6.2 SCD：自条件去噪

**核心思路**（Perez & Gómez-Bombarelli, 2026）：DeNS 的条件用的是**真实力**（需要标签）。
SCD 把它换成**模型自己对干净结构的嵌入**——同一个 batch 跑两次前向，
干净那次产出一个每构型一个的条件向量，注入到加噪那次里。于是整个训练**不需要任何标签**。

#### 6.2-A 条件向量的产生（双前向）

```
① clean 前向                                                    scd:209-248
   pos_clean（trainer 存的，或 pos - noise_vec）
     → generate_graph（借 data.pos 建图，用完立刻还回，不污染 batch）  scd:228-236
     → core_compute(fe=0 标量)  →  x_scalar [N, 128]              scd:241-247
② SCDCondHead：逐原子标量 → 每构型一个向量                          scd:26-51
   pre_proj (LN-Linear-SiLU-LN)          [N,128]
   index_add_ 按 batch 求和  ← 信息瓶颈    [B,128]
   post_proj (LN-Linear-SiLU-LN-Linear)  [B,128]
③ 条件后处理                                                     scd:261-280
   以 p=scd_p_dropcond 按图替换成可学的 mask_token（CFG 手法）      scd:261-265
   → scd_cond_norm(LayerNorm) → clamp(±scd_cond_clip)
   → scd_cond_proj(Linear, 零初始化)                              scd:144-145
   → c[data.batch]  广播回节点                    [N, 128]         scd:280
```

三个要点：

- **闸门**：`do_self_cond = self.training and data.denoising_pos_forward`（`scd:255`）。
  只有训练态 + 该 step 加了噪才走双前向；否则 `c = mask_token.expand(...)`。
  所以 `optim.use_denoising_pos: True` 是 SCD 生效的必要条件。
- **`scd_cond_proj` 零初始化**：初始状态下 SCD 分支恒输出 0，与父类 DeNS 逐位一致，可从 DeNS ckpt 无损续训。
- **`use_scd=False` 时不创建**这条链路（`scd:125`）——否则这些模块无人消费，变成 DDP unused parameter。

#### 6.2-B 两种注入方式

`scd_inject` 选 `input` / `adanorm` / `both`：

| 模式 | 注入点 | 实现 |
|---|---|---|
| `input` | 写进 L=0 输入嵌入，与 DeNS 的力条件同一入口 | `_scd_cond_embedding`（`scd:288-296`）→ 叠加到 `force_embedding`（`scd:323-326`） |
| `adanorm` | 每个 block 的两处 pre-norm | `_rebuild_blocks_with_adanorm`（`scd:189-207`）把 `norm_1`/`norm_2` 换成 `EquivariantAdaNorm` |

`input` 模式不改 `TransBlockV3` 签名，direct / gradient / compiled 三条前向路径原样复用；
`adanorm` 是论文形态，`cond` 作为显式入参一路传到 block（`dens:271-277` 的 `_forward_cond` 钩子，基类恒返回 `None`）。

#### 6.2-C EquivariantAdaNorm（`layer_norm.py:338-478`）

包住原有的等变 norm，由条件向量额外产出 `(shift, scale, gate)`：

```
cond[N,128] ─(use_node_feat 时 cat 上本节点 L=0 特征)─> fc  [in → 128 → out]   layer_norm:403-408
   fc = Linear-SiLU-LayerNorm-Linear，末层零初始化                            layer_norm:409-410
   out = C + 2·num_mod_degrees·C     (per_degree: num_mod_degrees = L+1 = 5)
                                      → L=4,C=128 时 out = 128 + 2·5·128 = 1408
split → shift[N,128] · scale[N,5,128] · gate[N,5,128]                        layer_norm:470-473

x = norm(x)                                    # 原 merge_layer_norm，fp32 守卫
x = x * (1 + broadcast(scale))                 # 逐 degree 展开到 25 分量      layer_norm:475
x = x + shift · l0_mask                        # shift 只进 L=0               layer_norm:477
return x, 1 + broadcast(gate)                  # gate 交给 block 乘在子层输出上  layer_norm:478
```

`_broadcast`（`layer_norm:446-452`）用 `expand_index` 把 `[N, L+1, C]` 展开成 `[N, 25, C]`，
**同一个 `l` 内的 `2l+1` 个 `m` 分量共享同一个系数**——这正是 §0 里等变性的要求：
Wigner-D 按 `l` 分块对角、只在块内混合 `m`，对 `m` 为常数的标量才能自由穿过。
`scope` 的三档就是调制粒度：`per_degree`（逐 `l`）/ `shared`（全 `l` 共享）/ `l0_only`（只调制 L=0）。

block 里的接线（`transformer_block.py:762-791`）：

```
outputs, gate_1 = norm_1(outputs, cond) → ga(...)  → outputs = outputs * gate_1 → +残差
outputs, gate_2 = norm_2(outputs, cond) → ffn(...) → outputs = outputs * gate_2 → +残差
```

**恒等初始化**：`fc` 末层零初始化 + `scale`/`gate` 以 `1 + Δ` 读出 ⇒ 未训练时本层与原 norm 逐位等价，
既有 ckpt 可无损续训（不同于 DiT 的 zero-init gate，那会让残差支在 step 0 完全关闭）。
配套有一个 `_load_from_state_dict` 钩子（`layer_norm:429-444`）把旧 ckpt 的 `affine_*` 重映射到 `norm.affine_*`。

> **ckpt 兼容性**：AdaNorm 多包了一层，键名从 `blocks.*.norm_1.affine_weight` 变成
> `blocks.*.norm_1.norm.affine_weight` 并新增 `blocks.*.norm_1.fc.*`。
> 因此 SCD 的 ckpt **不能**用裸 `equiformer_v3` / `equiformer_v3_dens` 加载（会被静默跳过），
> 且 `scd_adanorm_use_node_feat` / `scd_adanorm_scope` 变了会直接 shape 不匹配。

#### 6.2-D 元素嵌入冻结与漂移诊断

论文附录 B 报告：预训练不冻结元素嵌入会让其趋近于零，导致下游不稳定。
equiv3 的元素身份有**三个入口**共 `1 + 2 + 2N` 张表，故 `scd_freeze_element_embedding` 分四档
（`_apply_element_embedding_freeze`，`scd:156-187`）：

| 档位 | 新增冻结 | 对应入口 |
|---|---|---|
| `none` | — | — |
| `sphere` | `sphere_embedding` | 节点 L=0 嵌入（§2 ③a） |
| `sphere_edge` | ↑ + `edge_degree_embedding.{source,target}` | 边度嵌入的 m=0 径向系数（§2 ③b） |
| `all` | ↑ + 每层 `blocks[i].ga.{source,target}` | attention 内的边元素嵌入（§2 ④-A 步骤 1） |

输出头（`force_block`/`dens_block`/`stress_block`）明确不在冻结范围——它们是任务头而非输入通道。
冻结用的是 `requires_grad_(False)`（不是 `detach`）：参数不进优化器，**连权重衰减也一并跳过**。

trainer 侧配了两组诊断埋点，按 `print_every` 节流、只在 master rank 计算：
`element_embedding_norms`（`trainer:268-298`，三组表的 L2 范数 `emb_norm_*`）和
`element_embedding_drift`（`trainer:327-…`，相对训练起点的位移范数 `emb_drift_*`）。

### 6.3 三种模型前向对照

| | `equiformer_v3` | `+ DeNS` | `+ SCD` |
|---|---|---|---|
| 前向次数 / step | 1 | 1 | **2**（clean + noisy），非去噪 step 退回 1 |
| 输入侧条件 | — | 真实力 → 球谐 → `SO3Linear` | 自身对 clean 结构的嵌入（`input` 模式）|
| block 内注入 | — | — | AdaNorm 的 `shift/scale/gate`（`adanorm` 模式）|
| 新增输出头 | — | `dens_block`（一轮完整等变注意力）| 同左（复用父类）|
| `forces` 槽含义 | 力 | 按 `noise_mask` 混合力 / 噪声 | 同左 |
| 是否需要标签 | 需要 | 需要（力用作条件 + 未腐蚀原子的靶）| **不需要**（`use_force_cond=False` 时）|
| 主要额外开销 | 基准 | +1 次等变注意力 | +1 次完整前向，+ 每 block 2 个条件 MLP |

### 6.4 微调期的退化行为

下游监督微调设 `optim.use_denoising_pos: False`，此时：

- `denoising_pos_forward` 不存在 ⇒ `noise_mask` 全 0 ⇒ `outputs['forces']` 就是纯力，`dens_block` 输出被乘 0
  （仍前向、零梯度，DDP 安全）；
- `do_self_cond` 恒为 False ⇒ `cond` **恒等于 `mask_token`**（`scd:267`）——微调期它不是"用不到"，
  而是唯一的条件来源；
- 于是 AdaNorm 的调制退化成常量。特别地，`scd_adanorm_use_node_feat=False` 时它是**全数据集范围的一个常量**，
  可折叠进 norm 的 affine ⇒ 与普通 norm 属同一函数类（这使得"SCD 预训练 vs 随机初始化"的对照不会被
  "AdaNorm 本身是不是更强的 norm"污染）；`=True` 时调制仍逐节点变化，不可折叠。

> 保守力（`direct_prediction=False`）下另有一个约束：`use_node_feat=True` 时能量经调制系数依赖坐标，
> 而 `scd_adanorm_detach_node_feat` 默认 `True` 会截断这条反传路径，使 autograd 的 `-∂E/∂pos`
> 不是真实梯度（有限差分实测偏差远高于噪声地板）。保守力配置必须设 `False`，
> 或干脆用 `use_node_feat=False`（那样这条路径根本不存在）。direct 路径不受影响——力是独立输出头。

---

## 参考文件索引

- 主模型 / 前向：`experimental/models/equiformer_v3/equiformer_v3.py`
- DeNS 变体：`.../equiformer_v3_dens.py`；SCD 变体：`.../equiformer_v3_scd.py`
- DeNS/SCD trainer（加噪、换靶、诊断）：`experimental/trainers/equiformer_v3_dens_trainer.py`
- Wigner 旋转 / SO3Linear / SO3Grid：`.../so3.py`
- 欧拉角 / safe 反向：`.../edge_rot_mat.py`，`wigner_D`：`.../wigner.py`
- SO(2) 线性：`.../so2_ops.py`
- Transformer block / 注意力 / FFN：`.../transformer_block.py`
- 边度嵌入：`.../input_block.py`；输出头：`.../output_block.py`
- 等变归一化：`.../layer_norm.py`；径向函数：`.../radial_function.py`
- scatter 聚合：`.../utils.py`；包络：`.../envelope.py`；softmax：`.../softmax.py`
