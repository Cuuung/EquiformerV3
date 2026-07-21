# EquiformerV3 torch.compile 接入设计（Stage 4）

> 目标读者：本人 / 协作者。本 spec 是把 eSEN 已验证的 compile 工程（`ESEN_COMPILE_ROADMAP.md`
> Stage 0–3）迁移到 EquiformerV3 的落地方案。所有 esen 侧结论以该 roadmap 为权威来源，本文只记
> **V3 特有的差异与接线**，不重复 roadmap 已证内容。
>
> 仓库：`/mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3`
> 分支：`compile-equiformer-v3`（从 `muon-optimizer` 切出）
> 蓝图来源：`/mnt/afs/home/maoruicong/esen/fairchem`（已落地 Stage 0–3）

---

## 0. 目标与边界

**目标**：让 EquiformerV3（DeNS 与非 DeNS 两个 backbone）的前向与"能量→力/应力"二阶反传能被
`torch.compile`(inductor) 吃下，在不改变训练数值结果的前提下靠 infra 提速。

**本轮范围（B 档：A + 守恒力 dynamic=False，结构上为 C 留口）**：
1. 移植模型无关 harness `compile_utils.py`（直接拷贝，零改动）。
2. `edge_rot_mat.py` 迁移到 UMA Euler 路径（make_fx 友好、去 e3nn rot_clip/debug 守卫）。
3. 抽 `core_compute`（纯 tensor 进出的计算体）。
4. **直接力**：`plain_compile` 包 core_compute + TF32。
5. **守恒力**：`CompiledForceRegion`(dynamic=False) 包 `pos(+disp)→core_compute→energy→autograd.grad`。
6. 与现有 `optim.use_compile` 协调（互斥，见 §5）。
7. 每步带 gate（移植 esen 的测试脚本）。

**不做（本轮范围外）**：dynamic=True 免重编译（C 档，结构留口但不交付）；改模型结构/精度；
bf16（roadmap §7，后续）；triton kernel。

---

## 1. 环境与运行方式（零重装）

- **复用** `/mnt/afs/home/maoruicong/esen/.venv-torch211`：torch 2.11.0+cu128 /
  torch_scatter 2.1.2+pt211cu128 / e3nn 0.6.0。已实测本 repo V3 模型在此 import 干净。
- 本 repo 的 `fairchem.core` 与 esen 是不同 fork，**不能共用同一 editable 安装**。改用
  **PYTHONPATH 覆盖**让 `import fairchem.core` 解析到本 repo，不动 venv 的 pip 状态、不破坏 esen：
  ```bash
  PYTHONPATH=$PWD/src:$PWD /mnt/afs/home/maoruicong/esen/.venv-torch211/bin/python <cmd>
  ```
- 交付物：`scripts/compile_env.sh` 固化该调用约定（测试/训练统一走它）。
- torch 2.11 让 dynamic=True（C 档）**免费可达**，但本轮按 dynamic=False 交付。

---

## 2. 文件改动清单（按依赖顺序）

| # | 文件 | 改动 | 复用来源 |
|---|---|---|---|
| 1 | **新增** `src/fairchem/core/common/compile_utils.py` | 直接拷贝 esen 的 396 行 harness（模型无关，一字不改） | esen 同名文件 |
| 2 | `experimental/models/equiformer_v3/edge_rot_mat.py` | 迁移 UMA Euler：`Safeacos`/`Safeatan2`+`init_edge_rot_euler_angles`+`eulers_to_wigner`；删 min-dist 守卫(`:17-20`)、`assert max<0.99`(`:65`)、rot_clip mask | esen `models/esen/common/rotation.py` |
| 3 | `equiformer_v3.py` + `equiformer_v3_dens.py` | 抽 `core_compute`；ctor 加 `enable_compile`/`compile_dynamic`；lazy `_compiled_core`/`_compiled_region` | esen `esen.py`/`esen_dens.py` |
| 4 | 同上 `_forward_direct` | 直接力：`enable_compile` 时用 `plain_compile` 包 core_compute | esen 直接力策略 |
| 5 | 同上 `_forward_gradient` | 守恒力：`enable_compile` 时用 `CompiledForceRegion` 包二阶反传 region | esen `_conservative_compiled_forward` |
| 6 | `experimental/trainers/equiformer_v3_dens_trainer.py` | `enable_compile` 与 `optim.use_compile` 互斥（见 §5）；沿用 `optimize_ddp=False` | roadmap §1.10 |
| 7 | config yml | 加 `model.backbone.enable_compile`/`compile_dynamic`，默认 False | esen config |
| 8 | `compile_dev/`（新增目录） | 移植 esen 的 gate 脚本（见 §6） | esen `compile_dev/` |
| 9 | `ESEN_COMPILE_ROADMAP.md` Stage 4 | 把 5 行 checklist 展开成本方案落地版（file:line 级 + gate） | 本 spec |

---

## 3. rotation 迁移（edge_rot_mat.py）

**现状 blocker**（`experimental/models/equiformer_v3/edge_rot_mat.py`）：
- `:17-20` `if torch.min(edge_vec_0_distance) < 0.0001:` 数据相关 debug 分支
- `:65` `assert torch.max(vec_dot) < 0.99`
- rot_clip 布尔掩码（`use_rotation_mask`，仅守恒力 `direct_prediction=False` 时启用）

**迁移方案**：照搬 esen 迁移后的 `rotation.py` 模式——
- `Safeacos`/`Safeatan2`：clamp-safe backward 的 autograd.Function（替代 rot_clip 的梯度稳定作用）
- `init_edge_rot_euler_angles(edge_distance_vec)`：`F.normalize(eps).clamp(-1,1)` →
  `beta=Safeacos(y)` / `alpha=Safeatan2(x,z)` / `gamma=rand_like(alpha)*2π`，返回 `(-γ,-β,-α)`
- `eulers_to_wigner` + 现有 `wigner_D(l,α,β,γ,Jd)`（V3 已有 `Jd.pt`，后端与 esen 相同）
- 删除两个 debug 守卫和 rot_clip（`F.normalize` 的 eps + Safe* 已保证数值安全；数据质量检查若要保留，
  放 eager 建图区，不放编译/trace 区）

**无损性依据**：esen 已在训好的 auto_grad checkpoint 上验证 UMA Euler vs e3nn 路径能量 ~3e-6、
力 ~6.5e-4（在模型自身 fp32 等变噪声内，roadmap §1.13）。V3 wigner 后端与 esen 同构 → 张量层 drop-in。

**调用点**：`equiformer_v3.py:672 _init_edge_rot_mat` 改用 Euler 路径；`so3.py::SO3Rotation` 的
`use_rotation_mask` 参数随 rot_clip 一并消除（或保留 no-op 兼容）。

---

## 4. core_compute 抽取与编译策略

### 4.1 core_compute 边界（以 `_forward_gradient` 为准，`_dens.py:348`）

- **留 eager**（数据相关控制流 / 建图）：displacement 设置（`:362-385`）、`generate_graph`
  （`:387-398`）、`atomic_numbers`/`source`/`target`（`:400-402`）、DeNS `force_embedding`
  （`_forward_dens_force_encoding`，含数据相关分支，`:406`）。
- **进编译区 core_compute**（纯 tensor 进出）：`_forward_edge`(`:404`) → `_forward_embedding`(`:405`)
  → `+force_embedding`(`:407`) → `_forward_blocks`(`:408`) → `energy_block`+index_add → energy(`:421-424`)。
- **二阶反传**：`autograd.grad([energy.sum()],[pos,displacement],create_graph)`(`:429`) 已在
  model `_forward_gradient` 内 → **天然 DDP 安全**（无需像 esen 那样挂独立 head）。

> 非 DeNS `equiformer_v3.py` 的 `_forward_gradient`(`:558`) 同构、无 force_embedding，core_compute
> 入参少一项；对称处理。

### 4.2 直接力策略（`_forward_direct`）

```python
compute = self.core_compute
if self.enable_compile:
    if self._compiled_core is None:
        from fairchem.core.common.compile_utils import plain_compile
        self._compiled_core = plain_compile(self.core_compute, dynamic=self.compile_dynamic)
    compute = self._compiled_core
# ... 用 compute(...) 取代原 inline 计算
```
直接力无 double backward，`plain_compile` 即可（esen 实测 1.26–1.79x）。

### 4.3 守恒力策略（`_forward_gradient`）

用 `CompiledForceRegion`(dynamic=False) 包整段二阶反传。**stale-bake guard（关键正确性坑）**：
每批变化张量必须作 core_fn **显式输入**，否则同形状下批复用缓存图里 trace 时烤死的连接/原子类型
（静默错，roadmap Stage 3）。V3 显式输入清单：
```
pos, displacement,                          # grad 目标
edge_index, source_atomic_numbers, target_atomic_numbers,
edge_distance, edge_distance_vec,           # → rotation
edge_envelope_weight, batch,
force_embedding                             # DeNS（非 DeNS 无此项）
```
闭包只留 live module 引用（参数原地更新已由 esen `probe_makefx_param_staleness` 证：make_fx
traced 图引用 live 参数张量，`optimizer.step()` 后无需重 trace）。DeNS 的 denoising 分支回退 eager。

> 与 HybridMuon 的关系：HybridMuon 是标准 `torch.optim.Optimizer`，原地更新参数，与上述"参数
> staleness 已排除"结论一致 → compile region 与 muon 优化器正交，无额外接线。

---

## 5. 与现有 `optim.use_compile` 的协调（互斥，不叠加）

**`optim.use_compile` 现状**（`equiformer_v3_dens_trainer.py:379-381`）：
```python
if self.config['optim'].get('use_compile', False):
    self.model = torch.compile(self.model, dynamic=True)   # 外层整包
    torch._dynamo.config.optimize_ddp = False
```
它把**整个 model** 用 `torch.compile` 包起来，dynamo 拦截 `model.forward` 整段。model.forward
（`equiformer_v3.py:663`）按 `direct_prediction` 分派到 `_forward_direct`/`_forward_gradient`。
→ 实际只对**直接力**有意义（守恒力撞 double-backward 墙）；= roadmap §1.10 实测的"V3 现有
use_compile 是 Phase 1 直接力，未解决守恒力"。

**冲突**：`use_compile` 包外层 model；我们的 make_fx 编译内层 region（`CompiledForceRegion`，内含已
make_fx-traced+inductor-compiled 的 GraphModule）。两者同开 → 外层 dynamo re-trace 撞进内层已编译图，
嵌套 compile/图分裂冲突。

**互斥规则**：
- 两个开关分层：`optim.use_compile`（trainer，外层整包，仅直接力，无 make_fx）；
  `model.backbone.enable_compile`（model 内，内层 region，直接力+守恒力，make_fx-capable）。
- trainer 接线：
  ```python
  enable_compile = self.config['model'].get('backbone', {}).get('enable_compile', False)
  if self.config['optim'].get('use_compile', False) and not enable_compile:
      self.model = torch.compile(self.model, dynamic=True)
      torch._dynamo.config.optimize_ddp = False
  elif enable_compile:
      torch._dynamo.config.optimize_ddp = False   # 兜底；内层 region 也会调 configure_dynamo_for_compile
  ```
- **fail-fast**：config 同时设两者 True → raise，提示二选一，绝不静默嵌套。
- `use_compile` 保留不动（向后兼容现有直接力 config）；`enable_compile` 默认 False → 现有训练零影响。

---

## 6. 测试 / gate（移植 esen，每步过了才进下一步）

把 esen `compile_dev/` 的脚本拷进本 repo `compile_dev/`，改成调 V3 的
`core_compute`/`CompiledForceRegion`。gate 判据沿用 esen 定论。

| Stage | gate 脚本（移植自 esen） | 通过标准 |
|---|---|---|
| rotation 迁移 | `verify_rotation_migration.py` | 旋转等变误差 + 与旧实现对齐：能量 <1e-5，力同量级噪声 |
| core_compute 重构 | `verify_*_refactor.py` | 抽取后能量/力/stress 与重构前逐值对齐 err≈0 |
| 直接力编译 | `test_stage_direct.py`（← `test_stage3_direct.py`） | 能量/力 err<1e-5 + 提速为正 |
| 守恒力编译（硬） | `test_stage_conservative_{dens,base}.py`（← `test_stage3_conservative_*`） | 同权重 `cos(g_eager,g_compiled)>0.999` + `‖gc‖/‖ge‖` 偏差<5% + compiled loss 下降 + 换体系大小重 trace 不崩 |
| V3 风格探针 | `probe_v3style_compile.py`（esen 已写，直接复用） | dynamo 行为符合预期 |

**gate 判据要点（roadmap 定论）**：守恒力编译正确性 = **同权重下梯度对齐**，**不是**比独立训练 loss
轨迹（CUDA 原子非确定性 + fp32 非位级 → run-to-run 乱跳，是错的 gate）。`strip_detach` 是最危险的
静默失效点（detach 切断二阶梯度，力对但权重无梯度且不报错）→ "force-loss 真降 / 权重拿到梯度"设硬 gate。

---

## 7. 为 C（dynamic=True）留口

- harness 已自带 dynamic 全套（`make_prime_graph_example`/`mark_dynamic`/`LOCKED_INDUCTOR_OPTIONS`/
  `CompiledForceRegion(dynamic=True)`），**无需再写**。
- 本轮留口动作：
  1. core_compute 输入做成 symbolic-safe（`shape[0]` 取代 `len()`；permute 后 `reshape` 取代 `view`；
     维度链接用 `co.reshape(ei.shape[1],-1)` 式写法——对 dynamic=False 恒等无损）。
  2. `CompiledForceRegion.__call__` 预留 `trace_example`+`dynamic_dims` 参数位（harness 已有）。
- C 档只需补：质数维 example（`make_prime_graph_example`）+ 逐个清符号坍缩点（einsum→matmul/bmm，
  functional 化），蓝图 = esen `compile_dev/probe_symbolic_*` + roadmap Stage 3 dynamic 小节。

---

## 8. 交付顺序（实现计划骨架）

1. 拷 `compile_utils.py` + 写 `scripts/compile_env.sh` + 烟测 import。
2. 迁移 `edge_rot_mat.py` → rotation 等变 gate。
3. 抽 `core_compute`（DeNS + 非 DeNS）→ 重构无损 gate。
4. 直接力 `plain_compile` 接线 + TF32 → 直接力 gate。
5. 守恒力 `CompiledForceRegion` 接线（DeNS + 非 DeNS）→ 守恒力硬 gate。
6. trainer 互斥接线 + config 开关 + fail-fast。
7. 移植 esen gate 脚本到 `compile_dev/`，跑全套。
8. 把落地版写进 `ESEN_COMPILE_ROADMAP.md` Stage 4。

每步独立可交付、过 gate 才进下一步。任一 gate 不过即止损、回溯。
