# wandb 指标单位与计算口径

本项目(EquiformerV3 / MPtrj)训练时上报到 wandb 的 energy / forces / stress
指标,其**物理单位**、**计算口径**(逐结构 vs 逐原子 vs 逐分量、free-atom 掩码、
是否加回 linear-ref、归一化/反归一化)都在这里集中说明,并给出**代码出处**,
避免每次读曲线都要重新推一遍量纲。

> 结论先行:三个 MAE 全部是**物理单位**,在**反归一化之后**计算。
> `energy_per_atom` = eV/atom,`forces` = eV/Å,`stress` = eV/Å³(9 分量完整 3×3)。

---

## 1. 一览表

| wandb 指标 | 单位 | 口径 | 层级 |
|---|---|---|---|
| `*/energy_mae` | **eV** | 逐结构**总能**,加回 linear-ref 的全能量 | system |
| `*/energy_per_atom_mae` | **eV/atom** | 逐结构总能误差 ÷ **free-atom 数** | system |
| `*/forces_mae` | **eV/Å** | **逐分量**(x/y/z),**仅 free atoms** | atom |
| `*/forces_cosine_similarity` | 无量纲 | **逐原子 3 维矢量**的余弦,仅 free atoms | atom |
| `*/stress_mae` | **eV/Å³** | **9 分量**(完整 3×3 张量),逐结构 | system |

`*` = `train/` 或 `val/` 前缀(`logger.py` 对每个 key 加 `{split}/` 前缀)。

**换算备忘**(盯图 / 对文献时用):

- 能量:文献常报 **meV/atom** → wandb 的 eV/atom **×1000**。例:0.022 eV/atom = 22 meV/atom。
- 应力:**1 eV/Å³ = 160.2 GPa** → 例:0.0019 eV/Å³ ≈ 0.31 GPa。
- 力:eV/Å 即社区通用单位,可直接横比。

---

## 2. 归一化 / 反归一化流(loss 与 metric 的口径差异)

数据集 `transforms` 里的 `normalizer`(rmsd=0.8124567446457275,E/F/S **共用**同一值,
eSEN 约定)和 `element_references`(仅 energy)**不会**在 load 时改写 `batch.*`;
它们被 trainer 加载为 `self.normalizers` / `self.elementrefs`,在需要时**即时**施加。
因此 `batch.energy` / `batch.forces` / `batch.stress` 始终是**原始物理量**。

**训练 loss**(`ocp_trainer.py::_compute_loss`)——在**归一化/去参考空间**比:

```
target = batch[name]
target = elementrefs[name].dereference(target, batch)   # energy: 减 linear-ref → residual
target = normalizers[name].norm(target)                 # 减 mean / 除 rmsd
loss   = coeff * fn(pred, target)                        # pred 也是模型在该空间的直接输出
```

**评估 metric**(`ocp_trainer.py::_compute_metrics`)——在**物理全量空间**比:

```
target = batch[name]                                    # 原始物理量,NOT dereferenced
out[name] = _denorm_preds(name, out[name], batch)       # pred: 反归一化 + 加回 linear-ref
evaluator.eval(out, target)
```

`_denorm_preds`(`ocp_trainer.py:241`):先 `normalizers[name](pred)`(乘 rmsd / 加 mean),
再 `elementrefs[name](pred, batch)`(**加回**线性参考)。

> **关键**:loss 在 residual/归一化空间,metric 在**全能量物理空间**。
> 所以 wandb 上的 MAE 是可直接与文献物理量对比的量,而 loss 的绝对值不是。

---

## 3. 逐指标细节

### 3.1 Energy

- 有**两个**指标:`energy_mae`(总能,eV,逐结构一个标量)和
  `energy_per_atom_mae`(eV/atom)。config `evaluation_metrics.energy: [mae, per_atom_mae]`。
- **在全能量上算,不是 residual**:metric 路径只对 **pred** 做 `_denorm_preds`
  (含加回 ref),target 保持原始全能量 → 两侧都在全能量空间。
- **per-atom 的分母是 free-atom 数**,不是总原子数:`_compute_metrics:388-392`
  把 `natoms` 重定义为 `natoms_free`,`per_atom_mae` 除以 `target["natoms"]`(=free)。
  MPtrj 通常无固定原子 → free = total,数值一致;但口径上是 free-atom。
  (`evaluator.py::per_atom_mae` = `|target-pred| / natoms`)
- linear-ref 在 per-atom MAE 中**会抵消**(pred、target 两侧参考项相同,作差消去),
  所以它衡量的是"相对基准的每原子能量误差"。

### 3.2 Forces

- `forces_mae` 用 `evaluator.py::mae` = `torch.abs(target - pred)`(**逐元素**),
  evaluator 对全部元素求均值 → 对 `(free_atoms × 3)` 个 x/y/z **分量**平均。单位 eV/Å。
- **仅 free atoms**:`_compute_metrics:399-404`,atom 级且 `eval_on_free_atoms: True`
  → `target = target[mask]`、`out = out[mask]`,`mask = (batch.fixed == 0)`。
- `forces_cosine_similarity` 不同:`torch.cosine_similarity`,是**逐原子 3 维矢量**的
  方向余弦(不是逐分量),同样仅 free atoms。

### 3.3 Stress

- **9 分量(完整 3×3 张量),不是 6-Voigt**。根源在 a2g:
  `atoms_to_graphs.py:252-253` 用 `atoms.get_stress(apply_constraint=False, voigt=False)`
  → ASE 返回 3×3 → `data.stress` 存为 (3,3) → metric 里 `target.view(batch_size, -1)`
  = (batch_size, **9**)。
  (注意:直接 `atoms.get_stress()` 默认 `voigt=True` 会给 6 个数,那是**采样口径**,
  不是训练/metric 口径。)
- 因张量对称,独立值实为 6 个,但 metric 对 **9 个 slot** 求平均(3 对非对角重复计入)——
  与"按 6 独立分量算"的聚合值会略有差别,横比文献时留意。
- **单位 eV/Å³**(ASE stress 约定),metric 已反归一化回物理量纲,**无额外缩放**。
- 训练侧:stress 在 loss 里被 `normalizers.norm`(÷rmsd 0.812)、且本身 ~1e-3 量级
  → 归一化后极小,**靠 loss 系数 s100 补偿**才在训练中活跃。
- config 文件名里的 **`dens-no-stress`** 指 **DeNS 去噪辅助任务不含 stress**,
  与主 stress 头无关:主头 `regress_stress: True`、`loss s100`、`outputs.stress` 全在,
  stress 头正常训练。model 侧(`output_block.py` 的 stress head)经 CG change-matrix +
  einsum 从 irreps(rank-0 ⊕ rank-2)映射成对称 3×3 输出,与 9 分量 target 对齐。

---

## 4. 代码出处速查

| 事项 | 文件:行 |
|---|---|
| split 前缀 `train/`、`val/` | `common/logger.py:246` |
| train 每步 log 的 dict(含 lr/epoch/step) | `trainers/ocp_trainer.py:181-202` |
| `grad_norm` 单独 log | `trainers/base_trainer.py:1077` |
| pred 反归一化 + 加回 ref | `trainers/ocp_trainer.py:241-251` |
| loss:target dereference + norm | `trainers/ocp_trainer.py:351-355` |
| metric:target 保持全量、free-atom 掩码 | `trainers/ocp_trainer.py:373-428` |
| `mae` / `per_atom_mae` / `cosine_similarity` | `modules/evaluator.py` |
| stress 存为 9 分量(voigt=False) | `preprocessing/atoms_to_graphs.py:251-256` |
| stress head(irreps→3×3) | `experimental/models/equiformer_v3/output_block.py` |

---

## 5. 实测样例(交叉验证单位量级)

**AdamW 基线**,config
`.../direct/equiformer_v3_N@2_L@2_C@64_..._epochs@15-bs@64x8-lr@2e-4-wd@1e-3_dens-no-stress_loss-e5-f10-s100.yml`,
run `2026-06-05-02-52-48-mptrj_direct_N@2_L@2_C@64_15ep`,**最终 val(ep15)**:

| 指标 | 值 | 单位 | 换算 |
|---|---|---|---|
| `val/energy_per_atom_mae` | 0.0221 | eV/atom | ≈ 22.1 meV/atom |
| `val/forces_mae` | 0.0475 | eV/Å | — |
| `val/stress_mae` | 0.00192 | eV/Å³ | ≈ 0.31 GPa |

(参考:`val/energy_mae` = 0.604 eV 总能;`val/forces_cosine_similarity` = 0.574。)
样例数值的量级与上述单位自洽:stress ~1e-3 eV/Å³ ↔ 亚-GPa,符合 MPtrj 近平衡构型。
