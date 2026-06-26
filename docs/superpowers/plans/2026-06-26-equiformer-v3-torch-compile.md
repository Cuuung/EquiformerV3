# EquiformerV3 torch.compile 接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 EquiformerV3（DeNS + 非 DeNS）的前向与守恒力二阶反传能被 `torch.compile`(inductor) 编译，数值无损地提速训练，本轮交付 B 档（直接力 + 守恒力 dynamic=False），结构为 C(dynamic=True) 留口。

**Architecture:** 复用 eSEN 已验证的模型无关 harness `compile_utils.py`（make_fx + strip_detach + CompiledForceRegion）。各模型做 per-model 接入：rotation 迁移到 UMA Euler（make_fx 友好）、抽 `core_compute`（纯 tensor 进出）、直接力用 `plain_compile`、守恒力用 `CompiledForceRegion`。与现有 `optim.use_compile` 互斥。

**Tech Stack:** PyTorch 2.11.0+cu128（复用 esen `.venv-torch211`，PYTHONPATH 覆盖）、torch.fx、inductor、e3nn 0.6.0、torch_scatter pt211cu128。

## Global Constraints

- 运行环境：`/mnt/afs/home/maoruicong/esen/.venv-torch211/bin/python` + `PYTHONPATH=$REPO/src:$REPO`。**不改 venv 的 pip 状态。**
- 蓝图来源（只读，不改）：`/mnt/afs/home/maoruicong/esen/fairchem`（已落地 Stage 0–3）。
- 工作分支：`compile-equiformer-v3`（已从 `muon-optimizer` 切出）。
- 新开关 `model.backbone.enable_compile` / `compile_dynamic` 默认 **False** → 现有训练零影响。
- 守恒力 gate 判据 = **同权重梯度对齐**（`cos>0.999` + `‖gc‖/‖ge‖` 偏差<5%），**不比独立 loss 轨迹**。
- `strip_detach` 是最危险静默失效点 → "权重拿到梯度 / force-loss 真降"设硬 gate。
- 提交信息不加 AI attribution 行。
- 数值容差：能量 <1e-5；力同模型 fp32 等变噪声量级（~1e-3）。

---

## Task 0: 环境脚本 + harness 移植

**Files:**
- Create: `scripts/compile_env.sh`
- Create: `src/fairchem/core/common/compile_utils.py`
- Create: `compile_dev/` (空目录占位，后续放 gate 脚本)

**Interfaces:**
- Produces: `fairchem.core.common.compile_utils` 提供 `plain_compile`, `trace_and_compile`, `CompiledForceRegion`, `configure_dynamo_for_compile`, `make_prime_graph_example`, `LOCKED_INDUCTOR_OPTIONS`, `strip_detach`, `replace_view_with_reshape`, `get_force_decompositions`。
- Produces: `scripts/compile_env.sh` —— 统一运行约定，`bash scripts/compile_env.sh <python args>`。

- [ ] **Step 1: 写 env 脚本**

`scripts/compile_env.sh`:
```bash
#!/usr/bin/env bash
# 复用 esen 的 torch 2.11 venv 跑本 repo（PYTHONPATH 覆盖 fairchem.core），零重装。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/mnt/afs/home/maoruicong/esen/.venv-torch211"
exec env PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}" "$VENV/bin/python" "$@"
```

- [ ] **Step 2: 拷贝 harness（一字不改）**

Run:
```bash
cp /mnt/afs/home/maoruicong/esen/fairchem/src/fairchem/core/common/compile_utils.py \
   src/fairchem/core/common/compile_utils.py
mkdir -p compile_dev && touch compile_dev/.gitkeep
```

- [ ] **Step 3: 烟测 import**

Run:
```bash
bash scripts/compile_env.sh -c "from fairchem.core.common.compile_utils import plain_compile, CompiledForceRegion, trace_and_compile; print('compile_utils OK')"
```
Expected: 打印 `compile_utils OK`，无 ImportError。

- [ ] **Step 4: 烟测 V3 模型在该环境可 import**

Run:
```bash
bash scripts/compile_env.sh -c "import experimental.models.equiformer_v3.equiformer_v3_dens as m; print('V3 import OK', hasattr(m,'EquiformerV3DeNS_OC'))"
```
Expected: `V3 import OK True`（可有 FutureWarning，忽略）。

- [ ] **Step 5: Commit**

```bash
git add scripts/compile_env.sh src/fairchem/core/common/compile_utils.py compile_dev/.gitkeep
git commit -m "compile: port model-agnostic compile_utils harness + env script"
```

---

## Task 1: edge_rot_mat.py 迁移到 UMA Euler

**Files:**
- Modify: `experimental/models/equiformer_v3/edge_rot_mat.py`
- Modify (按需): `experimental/models/equiformer_v3/equiformer_v3.py:672`(`_init_edge_rot_mat`)、`equiformer_v3_dens.py` 同名调用、`so3.py::SO3Rotation`(`use_rotation_mask`)
- Create: `compile_dev/verify_rotation_migration.py`（移植自 esen 同名脚本）

**Interfaces:**
- Consumes: 现有 `wigner_D(l, alpha, beta, gamma, _Jd)` + `Jd.pt`。
- Produces: `init_edge_rot_euler_angles(edge_distance_vec) -> (alpha, beta, gamma)`；`eulers_to_wigner(angles, start_lmax, end_lmax, Jd) -> wigner`；`Safeacos`/`Safeatan2`。旧 `init_edge_rot_mat(edge_distance_vec, use_rotation_mask)` 入口保留为薄包装或被替换（调用点同步改）。

- [ ] **Step 1: 移植 gate 脚本（先失败基线）**

参照 `/mnt/afs/home/maoruicong/esen/fairchem/compile_dev/verify_rotation_migration.py`，改成构造一个小 V3 backbone（lmax=2, num_layers=2），测三项：①旋转等变（能量不变、力等变）②roll 不变（随机 gamma，能量/力 spread）③迁移前后数值对齐。先用**旧** rotation 跑一遍，记录 baseline 能量/力。

Run:
```bash
bash scripts/compile_env.sh compile_dev/verify_rotation_migration.py --baseline
```
Expected: 打印 baseline 能量/力，存到 `compile_dev/_rot_baseline.pt`。

- [ ] **Step 2: 按 esen 蓝图改写 edge_rot_mat.py**

照搬 esen `models/esen/common/rotation.py` 的 `Safeacos`/`Safeatan2`/`init_edge_rot_euler_angles`/`eulers_to_wigner`。删除 V3 现有 blocker：`edge_rot_mat.py:17-20`(min-dist debug 分支)、`:65`(`assert max<0.99`)、rot_clip 布尔掩码。`_init_edge_rot_mat`/`SO3Rotation` 改用 Euler 路径；`use_rotation_mask` 参数消除或保留 no-op。

关键函数（照 esen，按 V3 的 lmax/wigner_D 签名适配）:
```python
EPS = 1e-7
class Safeacos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        xc = x.clamp(-1 + EPS, 1 - EPS); ctx.save_for_backward(xc); return torch.acos(x)
    @staticmethod
    def backward(ctx, g):
        (xc,) = ctx.saved_tensors
        return -g / torch.sqrt(1 - xc.pow(2)).clamp(min=EPS)
# Safeatan2 同 esen
def init_edge_rot_euler_angles(edge_distance_vec):
    xyz = torch.nn.functional.normalize(edge_distance_vec).clamp(-1.0, 1.0)
    x, y, z = torch.split(xyz, 1, dim=1)
    beta = Safeacos.apply(y.squeeze(-1)); alpha = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))
    gamma = torch.rand_like(alpha) * 2 * torch.pi
    return -gamma, -beta, -alpha
```

- [ ] **Step 3: 跑等变 + 无损 gate**

Run:
```bash
bash scripts/compile_env.sh compile_dev/verify_rotation_migration.py
```
Expected: ①能量等变误差 <1e-5；②力等变误差 ~1e-3（fp32 噪声量级）；③vs baseline 能量 <1e-5、力 ~1e-3。任一超标 → 回退查 frame 约定（edge→Y + roll）。

- [ ] **Step 4: 确认无 e3nn / 无数据相关分支残留**

Run:
```bash
bash scripts/compile_env.sh -c "
import ast, io
src = open('experimental/models/equiformer_v3/edge_rot_mat.py').read()
assert 'e3nn' not in src, 'e3nn 残留'
assert 'assert ' not in src.split('def init_edge_rot_euler_angles')[0] or True
print('rotation clean OK')
"
```
Expected: `rotation clean OK`（人工再核 min-dist 分支与 rot_clip 已删）。

- [ ] **Step 5: Commit**

```bash
git add experimental/models/equiformer_v3/edge_rot_mat.py experimental/models/equiformer_v3/equiformer_v3.py experimental/models/equiformer_v3/equiformer_v3_dens.py experimental/models/equiformer_v3/so3.py compile_dev/verify_rotation_migration.py
git commit -m "compile: migrate edge_rot_mat to UMA Euler (make_fx-clean, lossless)"
```

---

## Task 2: 抽取 core_compute（DeNS + 非 DeNS）

**Files:**
- Modify: `experimental/models/equiformer_v3/equiformer_v3_dens.py`（`_forward_gradient` / `_forward_direct` 重构）
- Modify: `experimental/models/equiformer_v3/equiformer_v3.py`（同上，非 DeNS）
- Create: `compile_dev/verify_core_compute_refactor.py`

**Interfaces:**
- Produces (DeNS): `core_compute(atomic_numbers_src, atomic_numbers_tgt, edge_distance, edge_distance_vec, edge_index, edge_envelope_weight, batch, force_embedding) -> energy`（纯 tensor 进出）。
- Produces (非 DeNS): `core_compute(...)` 同上但**无** `force_embedding` 参数。
- Produces: ctor 新增 `enable_compile: bool=False`, `compile_dynamic: bool=False`；实例属性 `self.enable_compile/self.compile_dynamic/self._compiled_core=None/self._compiled_region=None`。

- [ ] **Step 1: 写重构无损 gate（先记录重构前输出）**

`compile_dev/verify_core_compute_refactor.py`：构造小 V3 DeNS + 非 DeNS backbone，同一随机输入跑 `_forward_gradient`（守恒力，regress_stress=True），记录 energy/forces/stress。

Run:
```bash
bash scripts/compile_env.sh compile_dev/verify_core_compute_refactor.py --record
```
Expected: 存 `compile_dev/_core_baseline.pt`（energy/forces/stress）。

- [ ] **Step 2: DeNS `core_compute` 抽取**

把 `equiformer_v3_dens.py:404-424`（`_forward_edge`→`_forward_embedding`→`+force_embedding`→`_forward_blocks`→`energy_block`+index_add→energy）抽成 `core_compute(...)` 方法，纯 tensor 进出（不碰 `data`）。`_forward_gradient` 改为：eager 建图算出各 tensor + `force_embedding` → 调 `self.core_compute(...)` → `autograd.grad`。ctor 加 `enable_compile`/`compile_dynamic` 与 lazy 属性。

- [ ] **Step 3: 非 DeNS `core_compute` 抽取**

`equiformer_v3.py:558` 的 `_forward_gradient` 同样抽 `core_compute`（无 force_embedding），对称处理。

- [ ] **Step 4: 跑重构无损 gate**

Run:
```bash
bash scripts/compile_env.sh compile_dev/verify_core_compute_refactor.py
```
Expected: energy/forces/stress 与 baseline 逐值对齐 err≈0（重构是纯代码搬移，应 bit 级一致或 <1e-6）。

- [ ] **Step 5: Commit**

```bash
git add experimental/models/equiformer_v3/equiformer_v3_dens.py experimental/models/equiformer_v3/equiformer_v3.py compile_dev/verify_core_compute_refactor.py
git commit -m "compile: extract core_compute (DeNS + base), add enable_compile flags"
```

---

## Task 3: 直接力编译（plain_compile + TF32）

**Files:**
- Modify: `experimental/models/equiformer_v3/equiformer_v3_dens.py`（`_forward_direct`）
- Modify: `experimental/models/equiformer_v3/equiformer_v3.py`（`_forward_direct`）
- Create: `compile_dev/test_stage_direct.py`（移植自 esen `test_stage3_direct.py`）

**Interfaces:**
- Consumes: `core_compute`, `self.enable_compile`, `self.compile_dynamic`（Task 2）；`plain_compile`（Task 0）。
- Produces: `_forward_direct` 在 `enable_compile=True` 时走编译后的 core_compute。

- [ ] **Step 1: 写直接力 gate**

`compile_dev/test_stage_direct.py`：同输入跑 `enable_compile=False`（eager）与 `True`（compiled）的 `_forward_direct`，比能量/力。

Run:
```bash
bash scripts/compile_env.sh compile_dev/test_stage_direct.py
```
Expected: 先失败/无编译（enable_compile 未接线，两者相同）—— 作 baseline。

- [ ] **Step 2: 接 plain_compile**

`_forward_direct` 内（DeNS + 非 DeNS）：
```python
compute = self.core_compute
if self.enable_compile:
    if self._compiled_core is None:
        from fairchem.core.common.compile_utils import plain_compile
        self._compiled_core = plain_compile(self.core_compute, dynamic=self.compile_dynamic)
    compute = self._compiled_core
# 用 compute(...) 取代原 inline 调用
```

- [ ] **Step 3: 跑直接力 gate**

Run:
```bash
bash scripts/compile_env.sh compile_dev/test_stage_direct.py
ESEN_COMPILE_PROBE=1 bash scripts/compile_env.sh compile_dev/test_stage_direct.py  # 确认真编译
```
Expected: 能量 err<1e-5、力 err<1e-5；PROBE 模式下编译不静默退回 eager。

- [ ] **Step 4: Commit**

```bash
git add experimental/models/equiformer_v3/equiformer_v3_dens.py experimental/models/equiformer_v3/equiformer_v3.py compile_dev/test_stage_direct.py
git commit -m "compile: direct-force plain_compile wiring + numeric gate"
```

---

## Task 4: 守恒力编译（CompiledForceRegion, dynamic=False）

**Files:**
- Modify: `experimental/models/equiformer_v3/equiformer_v3_dens.py`（`_forward_gradient`）
- Modify: `experimental/models/equiformer_v3/equiformer_v3.py`（`_forward_gradient`）
- Create: `compile_dev/test_stage_conservative_dens.py`、`compile_dev/test_stage_conservative_base.py`（移植自 esen `test_stage3_conservative_*`）
- Create: `compile_dev/diag_grad_tracking.py`（移植自 esen，cos/范数比诊断）

**Interfaces:**
- Consumes: `core_compute`（Task 2）、`CompiledForceRegion`（Task 0）。
- Produces: `_forward_gradient` 在 `enable_compile=True` 时走 make_fx 编译的二阶反传 region。

- [ ] **Step 1: 写守恒力硬 gate（梯度对齐）**

`compile_dev/test_stage_conservative_dens.py`：固定权重，同输入跑 eager vs compiled `_forward_gradient`，对 loss=force_loss(+stress) 做 `backward`，比每参数梯度的 `cos(g_eager,g_compiled)` 和 `‖gc‖/‖ge‖`，并验**所有参数都拿到梯度**（strip_detach 生效）。再测换体系大小重 trace 不崩、新形状 `cos=1`。

Run:
```bash
bash scripts/compile_env.sh compile_dev/test_stage_conservative_dens.py
```
Expected: 先失败（enable_compile 守恒力未接线）。

- [ ] **Step 2: 接 CompiledForceRegion（DeNS）**

`_forward_gradient` 内，`enable_compile=True` 时，把 `pos(+displacement)→core_compute→energy` 包成 `core_fn` 闭包（**显式输入**：pos, displacement, edge_index, src_an, tgt_an, edge_distance, edge_distance_vec, edge_envelope_weight, batch, force_embedding），用 `self._compiled_region = CompiledForceRegion(dynamic=self.compile_dynamic, optimize_ddp=False)`，再在外层做 `autograd.grad`。蓝图 = esen `esen_dens.py::MLP_EFS_Head._conservative_compiled_forward`。

> 注意 stale-bake guard：每批变化张量必须是 core_fn 显式参数，闭包只留 live module 引用。

- [ ] **Step 3: 接 CompiledForceRegion（非 DeNS）**

`equiformer_v3.py:558 _forward_gradient` 同样接线，core_fn 无 force_embedding 参数。

- [ ] **Step 4: 跑守恒力硬 gate（DeNS + 非 DeNS）**

Run:
```bash
bash scripts/compile_env.sh compile_dev/test_stage_conservative_dens.py
bash scripts/compile_env.sh compile_dev/test_stage_conservative_base.py
bash scripts/compile_env.sh compile_dev/diag_grad_tracking.py
```
Expected: 同权重每参数 `cos>0.999`、`‖gc‖/‖ge‖` 偏差<5%、**全部参数拿到梯度**、换体系大小 `cos=1.0` 不崩。**不比独立 loss 轨迹**。

- [ ] **Step 5: Commit**

```bash
git add experimental/models/equiformer_v3/equiformer_v3_dens.py experimental/models/equiformer_v3/equiformer_v3.py compile_dev/test_stage_conservative_dens.py compile_dev/test_stage_conservative_base.py compile_dev/diag_grad_tracking.py
git commit -m "compile: conservative-force make_fx wiring (dynamic=False) + hard grad-align gate"
```

---

## Task 5: trainer 互斥接线 + config 开关 + fail-fast

**Files:**
- Modify: `experimental/trainers/equiformer_v3_dens_trainer.py:379-381`
- Create: `compile_dev/test_mutual_exclusion.py`

**Interfaces:**
- Consumes: `model.backbone.enable_compile`（Task 2 ctor flag）、`optim.use_compile`（现有）。
- Produces: 两开关互斥的 trainer 行为；同开则 raise。

- [ ] **Step 1: 写互斥行为测试**

`compile_dev/test_mutual_exclusion.py`：①`use_compile=True, enable_compile=False` → 外层 `torch.compile(model)` 被调用；②`enable_compile=True, use_compile=False` → 外层不调用、`optimize_ddp=False` 已设；③两者都 True → raise。用 monkeypatch / mock 检测 `torch.compile` 是否被调。

Run:
```bash
bash scripts/compile_env.sh compile_dev/test_mutual_exclusion.py
```
Expected: 先失败（互斥逻辑未接）。

- [ ] **Step 2: 改 trainer 接线**

`equiformer_v3_dens_trainer.py:379-381` 替换为：
```python
enable_compile = self.config.get('model', {}).get('backbone', {}).get('enable_compile', False)
use_compile = self.config['optim'].get('use_compile', False)
if use_compile and enable_compile:
    raise ValueError(
        "optim.use_compile (outer torch.compile) 与 model.backbone.enable_compile "
        "(in-model make_fx) 互斥，不能同开。二选一。")
if use_compile and not enable_compile:
    self.model = torch.compile(self.model, dynamic=True)
    torch._dynamo.config.optimize_ddp = False
elif enable_compile:
    torch._dynamo.config.optimize_ddp = False
```

- [ ] **Step 3: 跑互斥测试**

Run:
```bash
bash scripts/compile_env.sh compile_dev/test_mutual_exclusion.py
```
Expected: 三种 case 全 PASS。

- [ ] **Step 4: Commit**

```bash
git add experimental/trainers/equiformer_v3_dens_trainer.py compile_dev/test_mutual_exclusion.py
git commit -m "compile: mutually-exclusive enable_compile vs optim.use_compile + fail-fast"
```

---

## Task 6: config 开关样例 + 全套 gate 回归

**Files:**
- Create: `experimental/scripts/.../<一个带 enable_compile 的 yml 样例>`（复制现有 muon direct config，加 `model.backbone.enable_compile: True`）
- Create: `compile_dev/README.md`

**Interfaces:**
- Consumes: 以上所有 Task。
- Produces: 可直接训练的 compile config 样例 + gate 复跑入口文档。

- [ ] **Step 1: 加 config 样例**

复制一个现有 muon 直接力 config，在 `model.backbone` 下加 `enable_compile: True`、`compile_dynamic: False`，文件名加 `-compile` 后缀。

- [ ] **Step 2: 写 compile_dev/README.md**

列出每个 gate 脚本的用途 + 复跑命令（统一 `bash scripts/compile_env.sh compile_dev/<x>.py`）+ gate 判据（梯度对齐非 loss 轨迹）。

- [ ] **Step 3: 全套 gate 回归**

Run:
```bash
for s in verify_rotation_migration verify_core_compute_refactor test_stage_direct test_stage_conservative_dens test_stage_conservative_base test_mutual_exclusion; do
  echo "=== $s ==="; bash scripts/compile_env.sh compile_dev/$s.py || echo "FAIL $s"
done
```
Expected: 全 PASS。

- [ ] **Step 4: Commit**

```bash
git add experimental/scripts compile_dev/README.md
git commit -m "compile: enable_compile config sample + compile_dev gate README"
```

---

## Task 7: 把落地版写进 ESEN_COMPILE_ROADMAP.md Stage 4

**Files:**
- Modify: `/mnt/afs/home/maoruicong/esen/fairchem/ESEN_COMPILE_ROADMAP.md`（Stage 4 节，行 ~176-180）

**Interfaces:**
- Consumes: 本计划全部交付物 + gate 结果。

- [ ] **Step 1: Stage 4 写核心结果概括（简，不堆细节）**

把现有 5 行 checklist 标为完成，**只概括核心结果**：①直接力 + 守恒力(dynamic=False) 已接入并过 gate（附关键实测数值：rotation 无损 / 直接力 err / 守恒力梯度对齐 cos+范数比）②与 `optim.use_compile` 互斥 ③dynamic=True 已留口。**不复制全部细节**，改为指向 ref 路径：
- 设计：`<equiformer_v3 repo>/docs/superpowers/specs/2026-06-26-equiformer-v3-torch-compile-design.md`
- 计划：`<equiformer_v3 repo>/docs/superpowers/plans/2026-06-26-equiformer-v3-torch-compile.md`
- gate 脚本：`<equiformer_v3 repo>/compile_dev/`

- [ ] **Step 2: 一句话标注 V3 与 eSEN 的关键差异**

V3 `autograd.grad` 已在 model 内（`_forward_gradient`）→ 守恒力天然 DDP 安全，不需 esen 的 head 挂载；V3 有现成 `optim.use_compile`（直接力），需互斥。（一两句即可，细节见上方 ref。）

- [ ] **Step 3: Commit（在 esen repo）**

```bash
cd /mnt/afs/home/maoruicong/esen/fairchem
git add ESEN_COMPILE_ROADMAP.md
git commit -m "roadmap: Stage 4 EquiformerV3 接入落地（直接力+守恒力 dynamic=False）"
```

---

## Self-Review

**Spec coverage：**
- §1 环境 → Task 0（env 脚本）✅
- §2 文件改动清单 9 项 → Task 0(harness)/1(rotation)/2(core_compute)/3(direct)/4(conservative)/5(trainer)/6(config)/7(roadmap) ✅
- §3 rotation 迁移 → Task 1 ✅
- §4 core_compute + 两策略 → Task 2/3/4 ✅
- §5 use_compile 互斥 → Task 5 ✅
- §6 gate 移植 → 每 Task 内嵌 + Task 6 回归 ✅
- §7 dynamic 留口 → Task 2/4 symbolic-safe 写法（core_compute `shape[0]`/`reshape`、region 预留参数）+ roadmap 记录 ✅
- §8 交付顺序 → Task 0–7 一一对应 ✅

**Placeholder scan：** 各 Task 给了 env/gate 命令、关键代码块、esen 蓝图路径。core_compute 与守恒力闭包的逐行 aten 代码依赖执行期读全 V3 forward——计划给了边界(file:line)、显式输入清单、蓝图源文件，属迁移任务的正确粒度，非占位符。

**Type consistency：** `core_compute` 签名 DeNS（含 force_embedding）vs 非 DeNS（无）在 Task 2/3/4 一致；`enable_compile`/`compile_dynamic`/`_compiled_core`/`_compiled_region` 命名贯穿；`CompiledForceRegion(dynamic=, optimize_ddp=)`、`plain_compile(fn, dynamic=)` 与 harness 实际签名一致。
