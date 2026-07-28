# SCD v1 (AdaNorm 条件注入) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 SCD 的自条件向量经 AdaNorm 注入 EquiformerV3 每个 transformer block 的两处 pre-norm，实现论文形态的 SCD，并留出元素嵌入冻结与正则化的参数接口。

**Architecture:** 新增 `EquivariantAdaNorm` 包住现有等变 norm，由条件向量产出 per-degree 的 `(shift, scale, gate)`；`cond` 经显式签名从 `core_compute` 逐级透传到 `TransBlockV3`；模型类 `equiformer_v3_scd` 新增 `scd_inject: input|adanorm|both` 开关，复用 v0 已验证的条件生产链路（clean 前向 / DropCond / ProjHead）。

**Tech Stack:** PyTorch 2.11, fairchem registry, e3nn, torch_geometric；测试在 `equiformer_v3:compile` 镜像内跑，需 1 张 GPU。

## Global Constraints

- 分支：`scd-v1`。设计依据：`docs/superpowers/specs/2026-07-29-scd-v1-design.md`。
- 仓库绝对路径：`/mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3`
- 测试命令（下文统一记作 `RUN <script>`）：
  ```bash
  docker run --rm --gpus '"device=0"' \
    -v /mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3:/mnt/afs/home/maoruicong/LAM_understanding/repositories/equiformer_v3 \
    equiformer_v3:compile bash -lc "python <script>"
  ```
  镜像的 `PYTHONPATH`/`WORKDIR` 已指向该绝对路径，必须用同路径挂载。
- 等变性判定阈值：`< 1e-4`（v0 实测量级 1e-7）。编译-eager 数值一致：`< 1e-4`。
- **identity-init 是硬约束**：调制头未训练时 AdaNorm 必须逐位等价于原 norm，既有 equiv3/DeNS ckpt 可无损续训。所有 `scale`/`gate` 以 `1 + delta` 读出，`fc` 末层 zero-init。
- **等变性充要条件**：`scale`/`gate` 对同一 `(l, c)` 的全部 `2l+1` 个 m 分量必须是同一个标量；`shift` 只能加在 L=0。
- 不改 `weight_decay`（保持 1e-3）与腐蚀噪声 `std`（保持 0.025）。
- 注释用中文，与仓库现有风格一致；不加 AI 署名行。
- 每个 task 结束即 commit。

---

### Task 1: `EquivariantAdaNorm` 模块

**Files:**
- Create: `experimental/tests/test_equiformer_v3_adanorm.py`
- Modify: `experimental/models/equiformer_v3/layer_norm.py`（在文件末尾 `RMSNorm` 之后追加）

**Interfaces:**
- Consumes: `get_normalization_layer(norm_type, lmax, num_channels, eps, affine, normalization)`（`layer_norm.py:15`）
- Produces:
  - `EquivariantAdaNorm(norm_type, lmax, num_channels, cond_channels, scope='per_degree', use_node_feat=True, eps=1e-5, affine=True, normalization='component')`
  - `forward(x: [N, (lmax+1)**2, C], cond: [N, cond_channels] | None) -> (x_out: [N, (lmax+1)**2, C], gate: [N, (lmax+1)**2, C] | None)`
  - `cond is None` 时返回 `(self.norm(x), None)`
  - `scope in ('per_degree', 'shared', 'l0_only')`

- [ ] **Step 1: 写失败测试**

创建 `experimental/tests/test_equiformer_v3_adanorm.py`：

```python
"""EquivariantAdaNorm 单元测试：等变性 / identity-init / 三种 scope。

需要 GPU 与项目训练镜像，直接运行：
    python experimental/tests/test_equiformer_v3_adanorm.py
非 0 退出码即为失败。
"""
import sys
import torch

from fairchem.core.common.utils import setup_imports

setup_imports()

from fairchem.experimental.models.equiformer_v3.layer_norm import (
    EquivariantAdaNorm,
    get_normalization_layer,
)
from fairchem.experimental.models.equiformer_v3.so3 import SO3Rotation
from fairchem.experimental.models.equiformer_v3.wigner import wigner_D

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILURES.append(name)


LMAX, C, COND_C, N = 3, 16, 12, 7


def make(scope="per_degree", use_node_feat=True):
    torch.manual_seed(0)
    return EquivariantAdaNorm(
        "merge_layer_norm", lmax=LMAX, num_channels=C,
        cond_channels=COND_C, scope=scope, use_node_feat=use_node_feat,
    ).to(DEV)


def rand_inputs():
    torch.manual_seed(1)
    x = torch.randn(N, (LMAX + 1) ** 2, C, device=DEV)
    cond = torch.randn(N, COND_C, device=DEV)
    return x, cond


# 1. identity-init：未训练时等价于原 norm
m = make()
x, cond = rand_inputs()
plain = get_normalization_layer("merge_layer_norm", lmax=LMAX, num_channels=C).to(DEV)
plain.load_state_dict(m.norm.state_dict())
out, gate = m(x, cond)
ref = plain(x)
check("identity-init: 输出等于原 norm", torch.allclose(out, ref, atol=1e-6),
      f"max|d|={(out - ref).abs().max().item():.2e}")
check("identity-init: gate 恒为 1", torch.allclose(gate, torch.ones_like(gate), atol=1e-6))

# 2. cond=None 走无条件分支
out_n, gate_n = m(x, None)
check("cond=None: gate 为 None 且输出等于原 norm",
      gate_n is None and torch.allclose(out_n, ref, atol=1e-6))

# 3. 等变性：打破 zero-init 后仍成立
for scope in ("per_degree", "shared", "l0_only"):
    m = make(scope)
    with torch.no_grad():
        m.fc[-1].weight.normal_(0, 0.2)
        m.fc[-1].bias.normal_(0, 0.2)
    x, cond = rand_inputs()

    alpha = torch.tensor(0.3, device=DEV)
    beta = torch.tensor(-0.7, device=DEV)
    gamma = torch.tensor(1.1, device=DEV)
    D = torch.block_diag(*[
        wigner_D(l, alpha, beta, gamma).to(DEV) for l in range(LMAX + 1)
    ])                                                  # [(L+1)^2, (L+1)^2]

    out_a, gate_a = m(x, cond)
    out_b, gate_b = m(torch.einsum("ij,njc->nic", D, x), cond)
    err = (torch.einsum("ij,njc->nic", D, out_a) - out_b).abs().max().item()
    check(f"等变性 scope={scope}", err < 1e-4, f"max|d|={err:.2e}")

    # gate 必须是不变量：旋转输入后 gate 不变
    gerr = (gate_a - gate_b).abs().max().item()
    check(f"gate 旋转不变 scope={scope}", gerr < 1e-4, f"max|d|={gerr:.2e}")

# 4. l0_only 只动 L=0
m = make("l0_only")
with torch.no_grad():
    m.fc[-1].weight.normal_(0, 0.2)
    m.fc[-1].bias.normal_(0, 0.2)
x, cond = rand_inputs()
out, gate = m(x, cond)
plain = get_normalization_layer("merge_layer_norm", lmax=LMAX, num_channels=C).to(DEV)
plain.load_state_dict(m.norm.state_dict())
ref = plain(x)
check("l0_only: L>=1 不被调制",
      torch.allclose(out[:, 1:], ref[:, 1:], atol=1e-6))
check("l0_only: L>=1 的 gate 恒为 1",
      torch.allclose(gate[:, 1:], torch.ones_like(gate[:, 1:]), atol=1e-6))
check("l0_only: L=0 确实被调制",
      not torch.allclose(out[:, 0], ref[:, 0], atol=1e-6))

# 5. use_node_feat=False 时 fc 输入维度只有 cond_channels
m = make("per_degree", use_node_feat=False)
check("use_node_feat=False: fc 输入维 == cond_channels",
      m.fc[0].in_features == COND_C, f"{m.fc[0].in_features}")
out, gate = m(*rand_inputs())
check("use_node_feat=False: 前向可跑", out.shape == (N, (LMAX + 1) ** 2, C))

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

`RUN experimental/tests/test_equiformer_v3_adanorm.py`

Expected: `ImportError: cannot import name 'EquivariantAdaNorm'`

- [ ] **Step 3: 实现 `EquivariantAdaNorm`**

追加到 `experimental/models/equiformer_v3/layer_norm.py` 末尾：

```python
class EquivariantAdaNorm(torch.nn.Module):
    """等变特征上的 Adaptive LayerNorm（SCD v1）。

    包住现有的等变 norm，由条件向量额外产出 (shift, scale, gate)：
      - `scale` / `gate` 是不变标量，同一 (l, c) 内所有 m 分量共享同一个值
        （由 `expand_index` 保证），故乘上去仍等变；
      - `shift` 只加在 L=0，故加上去仍等变。

    `fc` 末层零初始化，且 `scale` / `gate` 以 `1 + delta` 读出 —— 未训练时本层
    逐位等价于原 norm，既有 ckpt 可无损续训（不同于 DiT 的 zero-init gate，
    那会让残差支在 step 0 完全关闭）。

    Args:
        scope: 'per_degree' 每个 degree 独立的 scale/gate；'shared' 全 degree 共享；
               'l0_only' 只调制 L=0（对齐参考实现 `dx*gate_x, dvec` 的严格形态）。
        use_node_feat: 是否把本节点的 L=0 特征（detach）拼进条件 MLP 的输入。
    """

    _SCOPES = ('per_degree', 'shared', 'l0_only')

    def __init__(
        self,
        norm_type,
        lmax,
        num_channels,
        cond_channels,
        scope='per_degree',
        use_node_feat=True,
        eps=1e-5,
        affine=True,
        normalization='component'
    ):
        super().__init__()
        assert scope in self._SCOPES, f"unknown scope: {scope}"
        self.lmax = lmax
        self.num_channels = num_channels
        self.cond_channels = cond_channels
        self.scope = scope
        self.use_node_feat = use_node_feat

        self.norm = get_normalization_layer(
            norm_type, lmax, num_channels, eps, affine, normalization
        )

        self.num_mod_degrees = (lmax + 1) if scope == 'per_degree' else 1
        out_channels = num_channels + 2 * self.num_mod_degrees * num_channels
        in_channels = cond_channels + (num_channels if use_node_feat else 0)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, num_channels),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(num_channels),
            torch.nn.Linear(num_channels, out_channels),
        )
        torch.nn.init.constant_(self.fc[-1].weight, 0.0)
        torch.nn.init.constant_(self.fc[-1].bias, 0.0)

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)

        l0_mask = torch.zeros(1, (lmax + 1) ** 2, 1)
        l0_mask[0, 0, 0] = 1.0
        self.register_buffer('l0_mask', l0_mask)

    def __repr__(self):
        return (f"{self.__class__.__name__}(lmax={self.lmax}, "
                f"num_channels={self.num_channels}, cond_channels={self.cond_channels}, "
                f"scope={self.scope}, use_node_feat={self.use_node_feat})")

    def _broadcast(self, v):
        """[N, num_mod_degrees, C] -> 可与 [N, (lmax+1)**2, C] 相乘的 delta。"""
        if self.scope == 'per_degree':
            return torch.index_select(v, dim=1, index=self.expand_index)
        if self.scope == 'shared':
            return v                                    # [N, 1, C]，靠广播
        return v * self.l0_mask.to(v.dtype)             # l0_only：只有 L=0 非零

    def forward(self, x, cond=None):
        x = self.norm(x)
        if cond is None:
            return x, None

        if self.use_node_feat:
            node_feat = x.narrow(1, 0, 1).squeeze(1).detach()
            inp = torch.cat([cond, node_feat], dim=-1)
        else:
            inp = cond

        out = self.fc(inp).to(x.dtype)
        c, d = self.num_channels, self.num_mod_degrees
        shift = out.narrow(1, 0, c)
        scale = out.narrow(1, c, d * c).view(-1, d, c)
        gate = out.narrow(1, c + d * c, d * c).view(-1, d, c)

        x = x * (1.0 + self._broadcast(scale))
        # shift 只进 L=0，out-of-place（避免 version-counter / make_fx 问题）
        x = x + shift.unsqueeze(1) * self.l0_mask.to(x.dtype)
        return x, 1.0 + self._broadcast(gate)
```

- [ ] **Step 4: 运行确认通过**

`RUN experimental/tests/test_equiformer_v3_adanorm.py`

Expected: `ALL PASS`（13 项）

- [ ] **Step 5: Commit**

```bash
git add experimental/models/equiformer_v3/layer_norm.py experimental/tests/test_equiformer_v3_adanorm.py
git commit -m "SCD v1: EquivariantAdaNorm 模块

per-degree 的 shift/scale/gate，shift 只进 L=0、scale/gate 经 expand_index
广播保证同一 (l,c) 内 m 共享同一标量。fc 末层 zero-init 且以 1+delta 读出，
未训练时逐位等价于原 norm，既有 ckpt 可无损续训。"
```

---

### Task 2: `TransBlockV3` 接受 `cond`

**Files:**
- Modify: `experimental/models/equiformer_v3/transformer_block.py`（`TransBlockV3.__init__` 约 623-713 行、`forward` 约 715-758 行）
- Modify: `experimental/tests/test_equiformer_v3_adanorm.py`（追加 block 级测试）

**Interfaces:**
- Consumes: `EquivariantAdaNorm`（Task 1）
- Produces:
  - `TransBlockV3(..., cond_channels=None, adanorm_targets=(), adanorm_scope='per_degree', adanorm_use_node_feat=True)`
  - `TransBlockV3.forward(x, source_atomic_numbers, target_atomic_numbers, edge_distance, edge_index, edge_envelope_weight=None, batch=None, cond=None)`
  - 属性 `use_adanorm_1` / `use_adanorm_2`（Python bool，编译期常量）

- [ ] **Step 1: 写失败测试**

追加到 `experimental/tests/test_equiformer_v3_adanorm.py` 的 `print()` 汇总之前：

```python
# =============================== TransBlockV3 ===============================
from fairchem.experimental.models.equiformer_v3.transformer_block import TransBlockV3
from fairchem.experimental.models.equiformer_v3.so3 import SO3Rotation as _SO3Rot

BL, BC, BCOND = 2, 16, 12


def make_block(targets=('attn', 'ffn')):
    torch.manual_seed(0)
    so3_rotation = _SO3Rot(BL, BL, use_rotation_mask=False).to(DEV)
    return TransBlockV3(
        num_in_channels=BC, attn_hidden_channels=8, num_heads=2,
        attn_alpha_channels=8, attn_value_channels=4, ffn_hidden_channels=16,
        num_out_channels=BC, lmax=BL, mmax=BL, so3_rotation=so3_rotation,
        attn_grid_resolution_list=[14, 8], ffn_grid_resolution_list=[14, 14],
        max_num_elements=32, edge_channels_list=[8, 8, 8],
        norm_type='merge_layer_norm', drop_path_rate=0.0,
        cond_channels=BCOND, adanorm_targets=targets,
    ).to(DEV)


b = make_block()
check("block: attn/ffn 两处都建成 AdaNorm", b.use_adanorm_1 and b.use_adanorm_2)
check("block: norm_1 是 EquivariantAdaNorm", isinstance(b.norm_1, EquivariantAdaNorm))
check("block: norm_2 是 EquivariantAdaNorm", isinstance(b.norm_2, EquivariantAdaNorm))

b_attn = make_block(targets=('attn',))
check("block: 只指定 attn 时 norm_2 退回普通 norm",
      b_attn.use_adanorm_1 and not b_attn.use_adanorm_2
      and not isinstance(b_attn.norm_2, EquivariantAdaNorm))

b_none = make_block(targets=())
check("block: targets 为空时两处都是普通 norm",
      not b_none.use_adanorm_1 and not b_none.use_adanorm_2)

# identity-init：带 cond 与不带 cond 的输出必须一致
torch.manual_seed(3)
NB = 6
xb = torch.randn(NB, (BL + 1) ** 2, BC, device=DEV)
ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], device=DEV)
edge_dist = torch.rand(ei.shape[1], 8, device=DEV)
san = torch.randint(1, 30, (ei.shape[1],), device=DEV)
tan = torch.randint(1, 30, (ei.shape[1],), device=DEV)
bidx = torch.zeros(NB, dtype=torch.long, device=DEV)
condb = torch.randn(NB, BCOND, device=DEV)
b.eval()
with torch.no_grad():
    eulers = torch.zeros(ei.shape[1], 3, device=DEV)
    b.ga.so3_rotation.set_wigner_from_eulers(eulers)
    o_cond = b(xb, san, tan, edge_dist, ei, None, bidx, condb)
    b.ga.so3_rotation.set_wigner_from_eulers(eulers)
    o_none = b(xb, san, tan, edge_dist, ei, None, bidx, None)
check("block identity-init: 有无 cond 输出一致",
      torch.allclose(o_cond, o_none, atol=1e-6),
      f"max|d|={(o_cond - o_none).abs().max().item():.2e}")
```

- [ ] **Step 2: 运行确认失败**

`RUN experimental/tests/test_equiformer_v3_adanorm.py`

Expected: `TypeError: __init__() got an unexpected keyword argument 'cond_channels'`

- [ ] **Step 3: 实现**

在 `transformer_block.py` 顶部的 `from .layer_norm import ...` 处加入 `EquivariantAdaNorm`。

`TransBlockV3.__init__` 参数表末尾（`ffn_drop=0.0` 之后）追加：

```python
        ffn_drop=0.0,
        cond_channels=None,
        adanorm_targets=(),
        adanorm_scope='per_degree',
        adanorm_use_node_feat=True
    ):
```

把 `self.norm_1 = get_normalization_layer(...)` 替换为：

```python
        self.use_adanorm_1 = ('attn' in adanorm_targets) and (cond_channels is not None)
        self.use_adanorm_2 = ('ffn' in adanorm_targets) and (cond_channels is not None)

        if self.use_adanorm_1:
            self.norm_1 = EquivariantAdaNorm(
                norm_type, lmax=lmax, num_channels=num_in_channels,
                cond_channels=cond_channels, scope=adanorm_scope,
                use_node_feat=adanorm_use_node_feat
            )
        else:
            self.norm_1 = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_in_channels)
```

把 `self.norm_2 = get_normalization_layer(...)` 替换为同构的 `use_adanorm_2` 分支。

`forward` 改为：

```python
    def forward(
        self,
        x,                          # torch.Tensor
        source_atomic_numbers,
        target_atomic_numbers,
        edge_distance,
        edge_index,
        edge_envelope_weight=None,  # for smooth cutoff
        batch=None,                 # for GraphDropPath
        cond=None                   # for SCD AdaNorm，[N, cond_channels]
    ):
        outputs = x
        x_res = x

        if self.use_adanorm_1:
            outputs, gate_1 = self.norm_1(outputs, cond)
        else:
            outputs, gate_1 = self.norm_1(outputs), None
        outputs = self.ga(
            outputs,
            source_atomic_numbers,
            target_atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight
        )
        if gate_1 is not None:
            outputs = outputs * gate_1

        if self.drop_path is not None:
            outputs = self.drop_path(outputs, batch)
        if self.proj_drop is not None:
            outputs = self.proj_drop(outputs)

        outputs = outputs + x_res

        x_res = outputs
        if self.use_adanorm_2:
            outputs, gate_2 = self.norm_2(outputs, cond)
        else:
            outputs, gate_2 = self.norm_2(outputs), None
        outputs = self.ffn(outputs)
        if gate_2 is not None:
            outputs = outputs * gate_2

        if self.drop_path is not None:
            outputs = self.drop_path(outputs, batch)
        if self.proj_drop is not None:
            outputs = self.proj_drop(outputs)

        if self.ffn_shortcut is not None:
            x_res = self.ffn_shortcut(x_res)

        outputs = outputs + x_res
```

（`forward` 的 `return outputs` 保持原样。）

- [ ] **Step 4: 运行确认通过**

`RUN experimental/tests/test_equiformer_v3_adanorm.py`

Expected: `ALL PASS`（18 项）

- [ ] **Step 5: 回归 —— 现有模型未受影响**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `ALL PASS`（24 项，v0 全部保持通过）

- [ ] **Step 6: Commit**

```bash
git add experimental/models/equiformer_v3/transformer_block.py experimental/tests/test_equiformer_v3_adanorm.py
git commit -m "SCD v1: TransBlockV3 支持 cond，norm_1/norm_2 可切 AdaNorm

adanorm_targets 未包含的位置退回普通 norm，行为与现有代码逐位一致；
use_adanorm_* 是 Python bool，dynamo 会特化，无运行时分支开销。"
```

---

### Task 3: `cond` 透传三条前向路径

**Files:**
- Modify: `experimental/models/equiformer_v3/equiformer_v3.py`（`_forward_blocks` 457-516、`core_compute` 517-549、`_forward_direct` 550-621、`_conservative_compiled_forward` 727-860）
- Modify: `experimental/models/equiformer_v3/equiformer_v3_dens.py`（`core_compute` 271-305、`_forward_direct` 308-380、`_forward_gradient` 383-500、`_conservative_compiled_forward` 503-712）

**Interfaces:**
- Consumes: `TransBlockV3.forward(..., cond=None)`（Task 2）
- Produces:
  - `EquiformerV3_OC._forward_blocks(x, source_atomic_numbers, target_atomic_numbers, edge_distance, edge_index, edge_envelope_weight, batch, cond=None)`
  - `EquiformerV3_OC.core_compute(atomic_numbers, edge_distance, edge_distance_vec, edge_index, batch, cond=None)`
  - `EquiformerV3DeNS_OC.core_compute(atomic_numbers, edge_distance, edge_distance_vec, edge_index, batch, force_embedding, cond=None)`
  - `EquiformerV3DeNS_OC._forward_cond(data) -> torch.Tensor | None`（基类返回 `None`，Task 4 在 SCD 子类覆写）
  - `EquiformerV3DeNS_OC._forward_dens_force_encoding(data, cond=None)`（新增第二形参，DeNS 基类忽略之）

- [ ] **Step 1: 写失败测试**

追加到 `experimental/tests/test_equiformer_v3_scd.py` 的 `print()` 汇总之前：

```python
# ========================= cond 透传（Task 3） =========================
import inspect
from fairchem.experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
from fairchem.experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC

for fn, name in [
    (EquiformerV3_OC._forward_blocks, "_forward_blocks"),
    (EquiformerV3_OC.core_compute, "EquiformerV3_OC.core_compute"),
    (EquiformerV3DeNS_OC.core_compute, "DeNS.core_compute"),
]:
    params = inspect.signature(fn).parameters
    check(f"{name} 有 cond 形参且默认 None",
          "cond" in params and params["cond"].default is None)

check("DeNS 有 _forward_cond 钩子且默认返回 None",
      EquiformerV3DeNS_OC._forward_cond(None, None) is None)

params = inspect.signature(EquiformerV3DeNS_OC._forward_dens_force_encoding).parameters
check("_forward_dens_force_encoding 接受 cond 形参",
      "cond" in params and params["cond"].default is None)

# gradient checkpointing 分支也必须透传 cond
m_ckpt = build(gradient_checkpointing_block_list=[1, 1])
m_ckpt.train()
with torch.no_grad():
    m_ckpt.scd_cond_proj.weight.normal_(0, 0.05)
o_ckpt = m_ckpt(add_noise(make_batch(seed=51)))
o_ckpt["energy"].sum().backward()
nz_ckpt = [
    bool(p.grad is not None and p.grad.abs().sum() > 0)
    for n, p in m_ckpt.named_parameters()
    if n.startswith("scd_") and n != "scd_mask_token"
]
check("gradient checkpointing 下 cond 正确透传且梯度回流",
      all(nz_ckpt), f"{sum(nz_ckpt)}/{len(nz_ckpt)}")
```

- [ ] **Step 2: 运行确认失败**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: 前几项 FAIL（`cond` 不在形参里），`_forward_cond` 抛 `AttributeError`

- [ ] **Step 3: 改 `equiformer_v3.py`**

`_forward_blocks` 签名末尾加 `cond=None`，并在**两个**分支都传：

```python
                if self.gradient_checkpointing_block_list[i] == 0:
                    x = self.blocks[i](
                        x,
                        source_atomic_numbers,
                        target_atomic_numbers,
                        edge_distance,
                        edge_index,
                        edge_envelope_weight,
                        batch,     # for GraphDropPath
                        cond,      # for SCD AdaNorm
                    )
                elif self.gradient_checkpointing_block_list[i] == 1:
                    x = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x,
                        source_atomic_numbers,
                        target_atomic_numbers,
                        edge_distance,
                        edge_index,
                        edge_envelope_weight,
                        batch,     # for GraphDropPath
                        cond,      # for SCD AdaNorm
                        use_reentrant=False
                    )
```

`core_compute` 签名末尾加 `cond=None`，`_forward_blocks(...)` 调用末尾加 `cond`。

`_forward_direct` 的 `compute(...)` 调用末尾加 `None`（基类无条件）。

`_conservative_compiled_forward` 的 `_energy` 加末位形参 `cond`，内部 `self.core_compute(an, ed, edv, ei, batch, cond)`；`core_fn_stress` / `core_fn_force` 各加末位形参 `cond`；args 元组末尾加 `None`。

- [ ] **Step 4: 改 `equiformer_v3_dens.py`**

`core_compute` 签名改为 `(self, atomic_numbers, edge_distance, edge_distance_vec, edge_index, batch, force_embedding, cond=None)`，`_forward_blocks(...)` 调用末尾加 `cond`。

新增钩子（放在 `core_compute` 之前）：

```python
    def _forward_cond(self, data):
        """SCD 的节点级条件向量，[N, C] 或 None。

        基类恒返回 None（DeNS 无自条件）；`EquiformerV3SCD_OC` 覆写。
        显式返回值而非模块状态，是为了让 `cond` 能作为编译区的显式入参。
        """
        return None
```

`_forward_dens_force_encoding` 签名改为 `(self, data, cond=None)`，函数体不变（DeNS 忽略 `cond`）。

三条前向各改两处 —— 以 `_forward_direct` 为例：

```python
        cond = self._forward_cond(data)
        force_embedding, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor = \
            self._forward_dens_force_encoding(data, cond)
        ...
        x_scalar, x, edge_distance, edge_envelope_weight = compute(
            atomic_numbers,
            edge_distance,
            edge_distance_vec,
            edge_index,
            data.batch,
            force_embedding,
            cond,
        )
```

`_forward_gradient` 同样两处。

`_conservative_compiled_forward`：`_energy` 与两个 `core_fn_*` 各加末位形参 `cond`；args 元组按 `cond` 是否为 `None` 构造，使 `None` 情形不进 traced 参数（make_fx 把默认值当常量烤进图，而模型配置在构造时已固定，一个进程只会出现一种形态）：

```python
        def _energy(pos_p, cell_p, an, ei, co, batch, fe, n_sys, cond):
            ...
            x_scalar, _x, _, _ = self.core_compute(an, ed, edv, ei, batch, fe, cond)
            ...

        def core_fn_stress(pos, disp, an, ei, co, cell, batch, fe, cond=None):
            ...

        def core_fn_force(pos, an, ei, co, cell, batch, fe, cond=None):
            ...

        # args / dynamic_dims 按 cond 是否存在构造
        base_force_args = (pos, an, ei, co, cell, batch, fe)
        force_args = base_force_args if cond is None else (*base_force_args, cond)
        base_force_dyn = [(pos, [0]), (an, [0]), (ei, [1]), (co, [0]),
                          (cell, [0]), (batch, [0]), (fe, [0])]
        force_dyn = base_force_dyn if cond is None else [*base_force_dyn, (cond, [0])]
```

stress 分支同构（`base_stress_args = (pos, disp, an, ei, co, cell, batch, fe)`，`dynamic_dims` 相应带上 `(disp, [0])`）。

`_prime` 在 `dynamic=True` 时也要追加一个 prime `cond`：

```python
    def _prime(stress):
        pe = make_prime_graph_example(pos.device, dtype, stress=stress)
        fe_p = torch.zeros(pe[0].shape[0], *fe.shape[1:], device=pos.device, dtype=dtype)
        if cond is None:
            return (*pe, fe_p)
        cond_p = torch.zeros(pe[0].shape[0], *cond.shape[1:], device=pos.device, dtype=dtype)
        return (*pe, fe_p, cond_p)
```

编译区外那次给 `dens_block` 用的 eager `core_compute` 调用也要带上 `cond`。

- [ ] **Step 5: 运行确认通过**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `ALL PASS`（30 项）

- [ ] **Step 6: Commit**

```bash
git add experimental/models/equiformer_v3/equiformer_v3.py experimental/models/equiformer_v3/equiformer_v3_dens.py experimental/tests/test_equiformer_v3_scd.py
git commit -m "SCD v1: cond 经显式签名透传到 blocks

_forward_blocks / core_compute 加 cond=None，两个 checkpointing 分支都传；
DeNS 三条前向经 _forward_cond 钩子取 cond（基类返回 None）；编译区把 cond
作为显式 traced 入参而非闭包捕获，避免 stale-bake。"
```

---

### Task 4: `scd_inject` 开关与 AdaNorm 下发

**Files:**
- Modify: `experimental/models/equiformer_v3/equiformer_v3.py`（抽出 `_build_block_config`，纯重构）
- Modify: `experimental/models/equiformer_v3/equiformer_v3_scd.py`
- Modify: `experimental/tests/test_equiformer_v3_scd.py`

**Interfaces:**
- Produces: `EquiformerV3_OC._build_block_config(i) -> dict`（父类工厂方法，子类复用）
- Consumes: `_forward_cond` 钩子、`TransBlockV3(cond_channels=...)`、`EquivariantAdaNorm`
- Produces:
  - `EquiformerV3SCD_OC(..., scd_inject='adanorm', scd_adanorm_targets=('attn', 'ffn'), scd_adanorm_scope='per_degree', scd_adanorm_use_node_feat=True, ...)`
  - `_scd_cond_vector(data) -> torch.Tensor | None`（`[N, C]` 节点级条件，v0 `_scd_cond_embedding` 的前半段拆出）

- [ ] **Step 1: 写失败测试**

追加到 `experimental/tests/test_equiformer_v3_scd.py`：

```python
# ========================= scd_inject（Task 4） =========================
for mode in ("input", "adanorm", "both"):
    mm = build(scd_inject=mode)
    mm.train()
    with torch.no_grad():
        mm.scd_cond_proj.weight.normal_(0, 0.05)
        for blk in mm.blocks:
            if getattr(blk, "use_adanorm_1", False):
                blk.norm_1.fc[-1].weight.normal_(0, 0.05)
                blk.norm_2.fc[-1].weight.normal_(0, 0.05)
    o = mm(add_noise(make_batch(seed=60)))
    o["energy"].sum().backward()
    check(f"scd_inject={mode}: direct 前向+反传可跑", o["energy"].shape == (2,))

    mg = build(scd_inject=mode, direct_prediction=False)
    mg.train()
    og = mg(add_noise(make_batch(seed=61)))
    check(f"scd_inject={mode}: 保守力前向可跑", og["forces"].shape == (12, 3))

m_ada = build(scd_inject="adanorm")
check("scd_inject=adanorm: blocks 建成 AdaNorm",
      all(b.use_adanorm_1 and b.use_adanorm_2 for b in m_ada.blocks))
m_in = build(scd_inject="input")
check("scd_inject=input: blocks 不建 AdaNorm",
      all(not b.use_adanorm_1 and not b.use_adanorm_2 for b in m_in.blocks))

# identity-init：adanorm 模型未训练调制头时等价于 DeNS
torch.manual_seed(77)
m_scd = build(scd_inject="adanorm")
m_dens_cfg = dict(MODEL_CFG)
m_dens = registry.get_model_class("equiformer_v3_dens")(**m_dens_cfg).to(DEV)
shared = {k: v for k, v in m_scd.state_dict().items() if k in m_dens.state_dict()}
missing, unexpected = m_dens.load_state_dict(shared, strict=False)
m_scd.eval(); m_dens.eval()
b_a = add_noise(make_batch(seed=78))
b_b = add_noise(make_batch(seed=78))
b_b.pos = b_a.pos.clone(); b_b.pos_clean = b_a.pos_clean.clone()
b_b.noise_vec = b_a.noise_vec.clone()
with torch.no_grad():
    o_scd = m_scd(b_a)
    o_dens = m_dens(b_b)
de = (o_scd["energy"] - o_dens["energy"]).abs().max().item()
check("identity-init: adanorm 模型 == 同权重 DeNS", de < 1e-5, f"max|dE|={de:.2e}")

# adanorm 模式的旋转等变（调制头非零）
m_eq = build(scd_inject="adanorm")
with torch.no_grad():
    m_eq.scd_cond_proj.weight.normal_(0, 0.05)
    for blk in m_eq.blocks:
        blk.norm_1.fc[-1].weight.normal_(0, 0.05)
        blk.norm_1.fc[-1].bias.normal_(0, 0.05)
        blk.norm_2.fc[-1].weight.normal_(0, 0.05)
m_eq.eval()
ba = add_noise(make_batch(seed=79))
R = rot_mat(ba.pos.dtype)
bb = add_noise(make_batch(seed=79))
bb.pos = ba.pos @ R.T
bb.pos_clean = ba.pos_clean @ R.T
bb.noise_vec = ba.noise_vec @ R.T
bb.cell = torch.einsum("bij,kj->bik", ba.cell, R)
bb.forces = ba.forces @ R.T
with torch.no_grad():
    oa = m_eq(ba)
    ob = m_eq(bb)
e_err = (oa["energy"] - ob["energy"]).abs().max().item()
f_err = (oa["forces"] @ R.T - ob["forces"]).abs().max().item()
check("adanorm: 能量旋转不变", e_err < 1e-4, f"max|dE|={e_err:.2e}")
check("adanorm: 力旋转等变", f_err < 1e-4, f"max|dF|={f_err:.2e}")
```

- [ ] **Step 2: 运行确认失败**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `TypeError: __init__() got an unexpected keyword argument 'scd_inject'`

- [ ] **Step 3: 实现**

`equiformer_v3_scd.py` 顶部加 `from .transformer_block import TransBlockV3`。

`__init__` 参数表加：

```python
    def __init__(
        self,
        use_scd=True,
        use_force_cond=True,
        scd_inject='adanorm',
        scd_adanorm_targets=('attn', 'ffn'),
        scd_adanorm_scope='per_degree',
        scd_adanorm_use_node_feat=True,
        scd_p_dropcond=0.2,
        scd_cond_clip=100.0,
        scd_detach_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert scd_inject in ('input', 'adanorm', 'both'), f"unknown scd_inject: {scd_inject}"
        self.scd_inject = scd_inject
```

在 `self.apply(self._init_weights)` **之前**重建 blocks（AdaNorm 需要在权重初始化前建好）：

```python
        if self.use_scd and self.scd_inject in ('adanorm', 'both'):
            self._rebuild_blocks_with_adanorm(
                targets=tuple(scd_adanorm_targets),
                scope=scd_adanorm_scope,
                use_node_feat=scd_adanorm_use_node_feat,
            )
```

**先把父类的 block 配置抽成工厂方法**（避免在子类里逐字复制 32 个超参 —— 两处
参数表手动同步迟早脱节）。在 `equiformer_v3.py` 中，把 `__init__` 里
`# Transformer block` 之后的循环体（约 291-334 行）：

```python
        # Transformer block
        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if self.gradient_checkpointing_block_list[i] == 1:
                attn_activation = self.attn_activation.replace('_mem', '')
                ...
            block_config_dict = dict(...)
            block_class = TransBlockV3
            self.blocks.append(block_class(**block_config_dict))
```

改为：

```python
        # Transformer block
        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransBlockV3(**self._build_block_config(i)))
```

并新增方法（把原来的 `block_config_dict` 构造原样搬进来，**内容一字不改**，
只是把 `attn_activation` / `ffn_activation` 的分支一并纳入）：

```python
    def _build_block_config(self, i):
        """第 i 个 TransBlockV3 的构造参数。

        抽成方法以便子类（如 SCD 的 AdaNorm 变体）在同一份参数表上做增量，
        不必复制整张表。
        """
        if self.gradient_checkpointing_block_list[i] == 1:
            attn_activation = self.attn_activation.replace('_mem', '')
            ffn_activation  = self.ffn_activation.replace('_mem', '')
        else:
            attn_activation = self.attn_activation
            ffn_activation  = self.ffn_activation
        return dict(
            # …… 原 block_config_dict 的全部键值，原样搬入 ……
        )
```

> **实现者注意：** 这一步是**纯重构，不得有行为变化**。搬完先跑
> `RUN experimental/tests/test_equiformer_v3_scd.py`（v0 的 24 项）确认全绿，
> 再继续下面的 AdaNorm 部分。原循环里若还有 `attn_weights_drop` 之类的
> checkpointing 分支逻辑，一并搬进 `_build_block_config`。

然后在 SCD 子类新增：

```python
    def _rebuild_blocks_with_adanorm(self, targets, scope, use_node_feat):
        """在父类的 block 参数表上做增量，把两处 pre-norm 换成 AdaNorm。

        父类 `__init__` 已建好 `self.blocks`，这里按同一份配置重建并追加
        AdaNorm 相关参数。重建发生在 `self.apply(self._init_weights)` 之前，
        因此新模块同样会被正常初始化。
        """
        new_blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            cfg = self._build_block_config(i)
            cfg.update(
                cond_channels=self.num_channels,
                adanorm_targets=targets,
                adanorm_scope=scope,
                adanorm_use_node_feat=use_node_feat,
            )
            new_blocks.append(TransBlockV3(**cfg))
        self.blocks = new_blocks
```

把 v0 的 `_scd_cond_embedding` 拆成两段：

```python
    def _scd_cond_vector(self, data):
        """节点级条件向量 [N, C]，或 None（未启用 SCD）。"""
        if not self.use_scd:
            return None
        num_graphs = len(data.natoms)
        do_self_cond = self.training and getattr(data, "denoising_pos_forward", False)

        if do_self_cond:
            c = self._scd_clean_cond(data, num_graphs)
            if self.scd_detach_cond:
                c = c.detach()
            if self.scd_p_dropcond > 0.0:
                keep = (
                    torch.rand(num_graphs, device=c.device) >= self.scd_p_dropcond
                ).to(c.dtype).view(-1, 1)
                c = c * keep + self.scd_mask_token * (1.0 - keep)
        else:
            c = self.scd_mask_token.expand(num_graphs, -1)

        c = self.scd_cond_norm(c)
        c = c.clamp(min=-self.scd_cond_clip, max=self.scd_cond_clip)
        c = self.scd_cond_proj(c)
        return c[data.batch]

    def _forward_cond(self, data):
        """AdaNorm 用的 cond；inject 不含 adanorm 时返回 None。"""
        if self.scd_inject not in ('adanorm', 'both'):
            return None
        return self._scd_cond_vector(data)

    def _scd_cond_embedding(self, cond_nodes):
        """把节点级条件写进 L=0，得到可加到输入嵌入上的 [N, (lmax+1)^2, C]。"""
        cond_embedding = torch.zeros(
            (cond_nodes.shape[0], (self.lmax + 1) ** 2, self.num_channels),
            device=cond_nodes.device,
            dtype=cond_nodes.dtype,
        )
        cond_embedding[:, 0, :] = cond_nodes
        return cond_embedding
```

`_forward_dens_force_encoding` 改为接收 `cond` 并按 `scd_inject` 决定是否叠加：

```python
    def _forward_dens_force_encoding(self, data, cond=None):
        if self.use_force_cond:
            (
                force_embedding,
                noise_mask_tensor,
                dens_batch_mask_tensor,
                dens_mask_tensor,
            ) = super()._forward_dens_force_encoding(data)
        else:
            (
                _,
                _,
                noise_mask_tensor,
                dens_batch_mask_tensor,
                dens_mask_tensor,
            ) = self._generate_dens_data(data)
            force_embedding = torch.zeros((), dtype=self.dtype, device=self.device)
            noise_mask_tensor = noise_mask_tensor.view(-1, 1)

        if self.use_scd and self.scd_inject in ('input', 'both'):
            # adanorm 分支已经算过 cond，直接复用，避免第二次 clean 前向
            cond_nodes = cond if cond is not None else self._scd_cond_vector(data)
            force_embedding = force_embedding + self._scd_cond_embedding(cond_nodes)

        return (
            force_embedding,
            noise_mask_tensor,
            dens_batch_mask_tensor,
            dens_mask_tensor,
        )
```

- [ ] **Step 4: 修掉被签名变更打断的 v0 测试**

`_scd_cond_embedding` 的入参由 `data` 改成了 `cond_nodes`，而 v0 的
`experimental/tests/test_equiformer_v3_scd.py:111` 仍以 batch 调用它：

```python
cond = model._scd_cond_embedding(b)
```

改为走新的两段式接口（同时把该断言收紧到 `input` 模式，因为只有该模式才会
把条件写进输入嵌入）：

```python
model_in = build(scd_inject="input")
model_in.train()
model_in.dtype, model_in.device = b.pos.dtype, b.pos.device
cond = model_in._scd_cond_embedding(model_in._scd_cond_vector(b))
```

其后两条 `check(...)`（"零初始化下 cond 恒为 0"、"cond 只占 L=0 通道"）以及后续
用到 `model` 的语句保持不变。

- [ ] **Step 5: 运行确认通过**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `ALL PASS`（41 项）

- [ ] **Step 6: Commit**

```bash
git add experimental/models/equiformer_v3/equiformer_v3_scd.py experimental/tests/test_equiformer_v3_scd.py
git commit -m "SCD v1: scd_inject 开关（input|adanorm|both）

条件向量生产链路（clean 前向/dropcond/ProjHead）在三种模式间共享，
both 模式只算一次 cond。adanorm 模式下按父类 block_config_dict 重建
blocks，把 norm_1/norm_2 换成 EquivariantAdaNorm。"
```

---

### Task 5: 元素嵌入冻结四档

**Files:**
- Modify: `experimental/models/equiformer_v3/equiformer_v3_scd.py`
- Modify: `experimental/tests/test_equiformer_v3_scd.py`

**Interfaces:**
- Produces: `EquiformerV3SCD_OC(..., scd_freeze_element_embedding='none', scd_freeze_mask_token=False)`；`_apply_element_embedding_freeze()` 在 `__init__` 末尾调用

- [ ] **Step 1: 写失败测试**

追加到 `experimental/tests/test_equiformer_v3_scd.py`：

```python
# ===================== 元素嵌入冻结四档（Task 5） =====================
# MODEL_CFG: num_channels=32, edge_channels=32, max_num_elements=128, num_layers=2
#   sphere        = 128*32                     = 4096
#   edge_degree   = 2*128*32                   = 8192   -> 累计 12288
#   blocks(2 层)  = 2*2*128*32                 = 16384  -> 累计 28672
EXPECTED_FROZEN = {"none": 0, "sphere": 4096, "sphere_edge": 12288, "all": 28672}

for level, expected in EXPECTED_FROZEN.items():
    mf = build(scd_freeze_element_embedding=level)
    frozen = sum(p.numel() for p in mf.parameters() if not p.requires_grad)
    check(f"冻结档 {level}: 冻结参数数 == {expected}", frozen == expected, f"实际 {frozen}")

# 冻结后仍能前向+反传，且所有 requires_grad=True 的参数都进图
mf = build(scd_freeze_element_embedding="all")
mf.train()
with torch.no_grad():
    mf.scd_cond_proj.weight.normal_(0, 0.05)
of = mf(add_noise(make_batch(seed=70)))
of["energy"].sum().backward()
no_grad_names = [
    n for n, p in mf.named_parameters() if p.requires_grad and p.grad is None
]
check("冻结 all 后 DDP unused-param 安全", not no_grad_names, f"{no_grad_names[:3]}")

mt = build(scd_freeze_mask_token=True)
check("scd_freeze_mask_token=True 冻住 mask token",
      not mt.scd_mask_token.requires_grad)
```

> **实现者注意：** `EXPECTED_FROZEN` 的数值依赖 `MODEL_CFG` 的 `num_channels` / `edge_channels` / `max_num_elements` / `num_layers`。落地时先打印实际 shape 核对：`sphere_embedding.weight.numel()`、`edge_degree_embedding.source_embedding.weight.numel()`、`blocks[0].ga.source_embedding.weight.numel()`，再把常量写死。**不要为了让测试通过而反过来改期望值** —— 若与推算不符，先查是不是冻错了模块。

- [ ] **Step 2: 运行确认失败**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `TypeError: unexpected keyword argument 'scd_freeze_element_embedding'`

- [ ] **Step 3: 实现**

`__init__` 参数表加 `scd_freeze_element_embedding='none'`、`scd_freeze_mask_token=False`，并在 `__init__` 末尾（zero-init 之后）调用 `self._apply_element_embedding_freeze()`。

```python
    _FREEZE_LEVELS = ('none', 'sphere', 'sphere_edge', 'all')

    def _apply_element_embedding_freeze(self):
        """按档位冻结输入侧元素嵌入。

        论文附录 B：预训练不冻结元素嵌入会让其趋近于零，导致下游不稳定。
        equiv3 的元素身份有三个入口（sphere / edge-degree / 每个 attention
        block），故分四档。**输出头（force/dens/stress block）不在冻结范围内**
        —— 它们是任务头而非输入通道，SSL 预训练时 dens_block 正是被训练的头。
        """
        level = self.scd_freeze_element_embedding
        assert level in self._FREEZE_LEVELS, f"unknown freeze level: {level}"

        if self.scd_freeze_mask_token:
            self.scd_mask_token.requires_grad_(False)

        if level == 'none':
            return

        self.sphere_embedding.weight.requires_grad_(False)
        if level == 'sphere':
            return

        for emb in (self.edge_degree_embedding.source_embedding,
                    self.edge_degree_embedding.target_embedding):
            if emb is not None:
                emb.weight.requires_grad_(False)
        if level == 'sphere_edge':
            return

        for block in self.blocks:
            for emb in (block.ga.source_embedding, block.ga.target_embedding):
                if emb is not None:
                    emb.weight.requires_grad_(False)
```

- [ ] **Step 4: 运行确认通过**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `ALL PASS`（47 项）

- [ ] **Step 5: Commit**

```bash
git add experimental/models/equiformer_v3/equiformer_v3_scd.py experimental/tests/test_equiformer_v3_scd.py
git commit -m "SCD v1: 元素嵌入冻结四档

none/sphere/sphere_edge/all，输出头明确排除在冻结范围外。
muon.py:432 与 base_trainer.py:774 都跳过 requires_grad=False，
四档任意一档都不会让优化器分组或 DDP 出问题。"
```

---

### Task 6: clean 前向的正则化噪声

**Files:**
- Modify: `experimental/models/equiformer_v3/equiformer_v3_scd.py`（`_scd_clean_cond`）
- Modify: `experimental/tests/test_equiformer_v3_scd.py`

**Interfaces:**
- Produces: `EquiformerV3SCD_OC(..., scd_reg_noise_std=0.0)`

- [ ] **Step 1: 写失败测试**

```python
# ================= clean 前向正则化噪声（Task 6） =================
# 默认 0 时 clean 坐标必须保持不变（reg noise 是 out-of-place 的，不污染 batch）
m0 = build(scd_reg_noise_std=0.0)
m0.train()
b0 = add_noise(make_batch(seed=81))
before = b0.pos_clean.clone()
m0(b0)
check("scd_reg_noise_std=0: pos_clean 不被修改",
      torch.equal(b0.pos_clean, before))

# reg noise 开启时 pos_clean 同样不该被就地改写
mr0 = build(scd_reg_noise_std=0.05)
mr0.train()
br0 = add_noise(make_batch(seed=83))
before_r = br0.pos_clean.clone()
mr0(br0)
check("scd_reg_noise_std>0: pos_clean 仍不被就地改写",
      torch.equal(br0.pos_clean, before_r))

# 关掉 dropcond 以隔离变量：条件向量的差异必须只来自 reg noise
bq = add_noise(make_batch(seed=82))
m_on = build(scd_reg_noise_std=0.05, scd_p_dropcond=0.0)
m_off = build(scd_reg_noise_std=0.0, scd_p_dropcond=0.0)
m_on.train()
m_off.train()

torch.manual_seed(1)
a1 = m_on._scd_cond_vector(bq)
torch.manual_seed(2)
a2 = m_on._scd_cond_vector(bq)
check("reg noise 开: 两次 clean 前向的条件不同",
      not torch.allclose(a1, a2, atol=1e-6))

torch.manual_seed(1)
c1 = m_off._scd_cond_vector(bq)
torch.manual_seed(2)
c2 = m_off._scd_cond_vector(bq)
check("对照组 reg noise 关: 两次 clean 前向的条件相同",
      torch.allclose(c1, c2, atol=1e-6))
```

> **实现者注意：** 上面的对照组是这一项的关键 —— 没有它，`scd_reg_noise_std`
> 实际未生效时测试同样会绿（dropcond 的随机性足以让两次结果不同）。
> 两个模型都必须 `scd_p_dropcond=0.0`。

- [ ] **Step 2: 运行确认失败**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `TypeError: unexpected keyword argument 'scd_reg_noise_std'`

- [ ] **Step 3: 实现**

`__init__` 加 `scd_reg_noise_std=0.0` 并存为 `self.scd_reg_noise_std`。

`_scd_clean_cond` 中取到 `pos_clean` 之后、换入 `data.pos` 之前插入：

```python
        pos_clean = (
            data.pos_clean
            if hasattr(data, "pos_clean")
            else data.pos - data.noise_vec
        )
        # 论文附录 A.1 的 regularizing noise（sigma ~ 0.005）：只加在未腐蚀视图上。
        # 注意参考实现仅在 noise_in_loader=False 分支施加，其周期材料配置
        # （pretrain_amp20.yaml, noise_in_loader=True）实际未启用，故默认 0。
        if self.scd_reg_noise_std > 0.0 and self.training:
            pos_clean = pos_clean + torch.randn_like(pos_clean) * self.scd_reg_noise_std
```

（`pos_clean` 是新张量，不写回 `data.pos_clean`，故不污染 batch。）

- [ ] **Step 4: 运行确认通过**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `ALL PASS`（52 项）

- [ ] **Step 5: Commit**

```bash
git add experimental/models/equiformer_v3/equiformer_v3_scd.py experimental/tests/test_equiformer_v3_scd.py
git commit -m "SCD v1: clean 前向的正则化噪声开关

默认 0：参考实现只在 noise_in_loader=False 分支施加，其周期材料配置
未启用，本项目场景是材料，无正面证据。out-of-place，不污染 batch。"
```

---

### Task 7: 元素嵌入范数诊断埋点

**Files:**
- Modify: `experimental/trainers/equiformer_v3_dens_trainer.py`
- Test: 见 Step 1

**Interfaces:**
- Produces: `EquiformerV3DeNSTrainer._log_element_embedding_norms() -> dict[str, float]`，在 `train()` 的既有 `self.log_dict(...)` 处按 `log_every` 节流调用

- [ ] **Step 1: 写失败测试**

创建 `experimental/tests/test_element_embedding_diag.py`：

```python
"""元素嵌入范数诊断埋点的单元测试。"""
import sys
import torch

from fairchem.core.common.utils import setup_imports
from fairchem.core.common.registry import registry

setup_imports()

from fairchem.experimental.trainers.equiformer_v3_dens_trainer import (
    element_embedding_norms,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILURES.append(name)


MODEL_CFG = dict(
    use_pbc=True, use_pbc_single=True, otf_graph=True,
    regress_forces=True, regress_stress=True, direct_prediction=True,
    max_neighbors=20, max_radius=5.0, num_radial_basis=10, max_num_elements=128,
    num_layers=2, num_channels=32, attn_hidden_channels=16, num_heads=4,
    attn_alpha_channels=16, attn_value_channels=8, ffn_hidden_channels=64,
    norm_type="merge_layer_norm", lmax=2, mmax=2,
    attn_grid_resolution_list=[14, 8], ffn_grid_resolution_list=[14, 14],
    edge_channels=32, drop_path_rate=0.0, attn_weights_drop=0.0,
    gradient_checkpointing_block_list=[0, 0], avg_num_nodes=1,
)

m = registry.get_model_class("equiformer_v3_scd")(**MODEL_CFG).to(DEV)
norms = element_embedding_norms(m)

check("返回三个键", set(norms) == {"emb_norm_sphere", "emb_norm_edge_degree", "emb_norm_blocks"},
      f"{sorted(norms)}")
check("全部为有限正数", all(0.0 < v < float("inf") for v in norms.values()), f"{norms}")

with torch.no_grad():
    m.sphere_embedding.weight.mul_(0.0)
norms2 = element_embedding_norms(m)
check("置零后 sphere 范数为 0", norms2["emb_norm_sphere"] == 0.0)
check("置零 sphere 不影响其余两项",
      norms2["emb_norm_edge_degree"] == norms["emb_norm_edge_degree"])

# 对不带 SCD 的普通 DeNS 模型也要能用
md = registry.get_model_class("equiformer_v3_dens")(**MODEL_CFG).to(DEV)
check("对 equiformer_v3_dens 同样可用", len(element_embedding_norms(md)) == 3)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

`RUN experimental/tests/test_element_embedding_diag.py`

Expected: `ImportError: cannot import name 'element_embedding_norms'`

- [ ] **Step 3: 实现**

在 `equiformer_v3_dens_trainer.py` 的 `compute_atomwise_denoising_pos_and_force_hybrid_loss` 之后追加模块级函数：

```python
def element_embedding_norms(model):
    """三组元素嵌入的 L2 范数，用于诊断预训练期的嵌入塌缩。

    equiv3 的元素身份有三个入口：`sphere_embedding`（节点级 L=0）、
    `EdgeDegreeEmbedding` 的 source/target（调制边的 m=0 径向系数）、
    以及每个 attention block 的 source/target。SCD 论文附录 B 报告不冻结
    元素嵌入会使其趋近于零；本函数让"是否需要冻结、冻到哪档"可由一次
    `scd_freeze_element_embedding=none` 的预训练直接读出。
    """
    m = model.module if hasattr(model, 'module') else model
    m = getattr(m, '_orig_mod', m)

    out = {'emb_norm_sphere': float(m.sphere_embedding.weight.norm())}

    edge_norms = [
        float(emb.weight.norm())
        for emb in (m.edge_degree_embedding.source_embedding,
                    m.edge_degree_embedding.target_embedding)
        if emb is not None
    ]
    out['emb_norm_edge_degree'] = sum(edge_norms) / len(edge_norms) if edge_norms else 0.0

    block_norms = [
        float(emb.weight.norm())
        for block in m.blocks
        for emb in (block.ga.source_embedding, block.ga.target_embedding)
        if emb is not None
    ]
    out['emb_norm_blocks'] = sum(block_norms) / len(block_norms) if block_norms else 0.0

    return out
```

在 `EquiformerV3DeNSTrainer.train()` 里，找到既有的 `self.log_dict(train_metrics, sync_dist=True)` 之前（即已按 `print_every` 节流的日志分支内），加入：

```python
                    train_metrics.update(element_embedding_norms(self.model))
```

> **实现者注意：** `train()` 中构造 `train_metrics` 并调 `log_dict` 的位置可能不止一处。只加在**已经按 `print_every` 节流**的那一处 —— 每步都算三组范数会拖慢训练。落地前先 `grep -n "log_dict" experimental/trainers/equiformer_v3_dens_trainer.py` 确认。

- [ ] **Step 4: 运行确认通过**

`RUN experimental/tests/test_element_embedding_diag.py`

Expected: `ALL PASS`（5 项）

- [ ] **Step 5: Commit**

```bash
git add experimental/trainers/equiformer_v3_dens_trainer.py experimental/tests/test_element_embedding_diag.py
git commit -m "SCD v1: 元素嵌入范数诊断埋点

三组入口（sphere/edge_degree/blocks）的 L2 范数写进 train_metrics，
一次 none 档预训练即可回答冻结档位的选择，避免盲试四档。"
```

---

### Task 8: eager/fp32 N2L2C64 配置与端到端验证

**Files:**
- Create: `experimental/configs/omat24/mptrj/experiments/direct/scd_v1/eager_fp32_N2L2C64.yml`
- Create: `experimental/tests/test_scd_v1_config.py`
- Modify: `docs/SCD_V0_NOTES.md` → 重命名为 `docs/SCD_NOTES.md` 并补充 v1

**Interfaces:**
- Consumes: Task 1-7 的全部产物

- [ ] **Step 1: 写失败测试**

创建 `experimental/tests/test_scd_v1_config.py`：

```python
"""SCD v1 配置的实例化与关键开关校验。"""
import copy
import sys
import yaml
import torch

from fairchem.core.common.utils import setup_imports
from fairchem.core.common.registry import registry

setup_imports()

P = ("experimental/configs/omat24/mptrj/experiments/direct/scd_v1/"
     "eager_fp32_N2L2C64.yml")
FAILURES = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILURES.append(name)


cfg = yaml.safe_load(open(P))
mc = copy.deepcopy(cfg["model"])
name = mc.pop("name")
m = registry.get_model_class(name)(**mc)

check("模型可实例化", name == "equiformer_v3_scd", f"{m.num_params/1e6:.2f}M")
check("scd_inject=adanorm", m.scd_inject == "adanorm")
check("blocks 全部建成 AdaNorm",
      all(b.use_adanorm_1 and b.use_adanorm_2 for b in m.blocks))
check("eager: optim.use_compile 为 False", cfg["optim"]["use_compile"] is False)
check("eager: model 未设 enable_compile", "enable_compile" not in cfg["model"])
check("drop_path_rate == 0.1", cfg["model"]["drop_path_rate"] == 0.1)
check("冻结档默认 none", m.scd_freeze_element_embedding == "none")
check("reg noise 默认 0.0", m.scd_reg_noise_std == 0.0)
check("N2L2C64 结构",
      cfg["model"]["num_layers"] == 2 and cfg["model"]["lmax"] == 2
      and cfg["model"]["num_channels"] == 64)
check("trainer 为 equiformer_v3_dens_trainer",
      cfg["trainer"] == "equiformer_v3_dens_trainer")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

`RUN experimental/tests/test_scd_v1_config.py`

Expected: `FileNotFoundError`

- [ ] **Step 3: 建配置**

```bash
mkdir -p experimental/configs/omat24/mptrj/experiments/direct/scd_v1
cp experimental/configs/omat24/mptrj/experiments/direct/compile_test/compile_fp32.yml \
   experimental/configs/omat24/mptrj/experiments/direct/scd_v1/eager_fp32_N2L2C64.yml
```

在新文件中做且仅做以下修改：

1. `model.name`: `equiformer_v3_dens` → `equiformer_v3_scd`，并在其下插入：

```yaml
model:
  name:                         equiformer_v3_scd

  # ---- SCD v1（自条件去噪，AdaNorm 注入）----
  use_scd:                      True
  use_force_cond:               True    # False = 论文形态的纯自条件
  scd_inject:                   adanorm # input | adanorm | both
  scd_adanorm_targets:          ['attn', 'ffn']
  scd_adanorm_scope:            per_degree   # per_degree | shared | l0_only
  scd_adanorm_use_node_feat:    True
  scd_p_dropcond:               0.2
  scd_cond_clip:                100.0
  scd_detach_cond:              False
  scd_reg_noise_std:            0.0     # 论文 0.005，但其材料配置未启用，故默认关
  scd_freeze_element_embedding: none    # none | sphere | sphere_edge | all
  scd_freeze_mask_token:        False
```

2. `model.drop_path_rate`: `0.05` → `0.1`，并加注释：

```yaml
  drop_path_rate:               0.1     # 论文 Table 17：双前向下 droppath 0.1 提升稳定性。
                                        # 注意 equiv3 的 GraphDropPath 是 per-graph 丢、
                                        # 参考实现是 per-node，同数值下本仓库正则更强 —— 第一待调项。
```

3. `optim.use_compile`: `True` → `False`，并把行尾注释改为：

```yaml
  use_compile:                  False   # v1 先验证正确性：纯 fp32 eager，不开 amp/tf32/compile
```

其余（数据集、normalizer、element_references、loss 权重、outputs、N2L2C64 结构、
HybridMuon/moonlight 参数、DeNS 噪声参数）全部保持模板原样，逐字不动。

- [ ] **Step 4: 运行确认通过**

`RUN experimental/tests/test_scd_v1_config.py`

Expected: `ALL PASS`（10 项）

- [ ] **Step 5: 全量回归**

依次运行三个测试文件：

```bash
RUN experimental/tests/test_equiformer_v3_adanorm.py
RUN experimental/tests/test_equiformer_v3_scd.py
RUN experimental/tests/test_element_embedding_diag.py
```

Expected: 三个都 `ALL PASS`（分别 18 / 52 / 5 项）

- [ ] **Step 6: 更新文档**

`git mv docs/SCD_V0_NOTES.md docs/SCD_NOTES.md`，把标题改为「EquiformerV3-SCD：自条件去噪」，
并在 §5「v0 未做的（留给 v1）」之后新增 v1 章节，覆盖：`scd_inject` 三种模式、
`EquivariantAdaNorm` 的等变性论证与 identity-init、冻结四档表、正则化默认值表、
新增的三个测试文件与运行方式。文末指向
`docs/superpowers/specs/2026-07-29-scd-v1-design.md`。

- [ ] **Step 7: Commit**

```bash
git add experimental/configs/omat24/mptrj/experiments/direct/scd_v1/eager_fp32_N2L2C64.yml \
        experimental/tests/test_scd_v1_config.py docs/SCD_NOTES.md
git rm --cached docs/SCD_V0_NOTES.md 2>/dev/null || true
git commit -m "SCD v1: eager/fp32 N2L2C64 验证配置 + 文档

纯 fp32 eager（不开 amp/tf32/compile），先验证正确性再谈性能。
drop_path_rate 0.05 -> 0.1（论文 Table 17，双前向）。"
```

---

### Task 9: 编译路径 × adanorm 验证

**前置：** Task 8 的 eager 验证已全绿。本 task 覆盖 spec §5.6。

**Files:**
- Modify: `experimental/tests/test_equiformer_v3_scd.py`（现有 A/B/C/D 四段编译测试）

**Interfaces:**
- Consumes: Task 3 接进两条编译路径的 `cond` traced 入参

- [ ] **Step 1: 把四段编译测试参数化到 adanorm**

现有 A/B/C/D 四段各自调用 `build(...)`。把每段的模型构造改为显式指定注入模式，
并对 `input` / `adanorm` 两种模式各跑一遍。以 A 段为例：

```python
for _mode in ("input", "adanorm"):
    try:
        torch._dynamo.reset()
        m = build(scd_inject=_mode)
        m.train()
        with torch.no_grad():
            m.scd_cond_proj.weight.normal_(0, 0.05)
            for blk in m.blocks:
                if getattr(blk, "use_adanorm_1", False):
                    blk.norm_1.fc[-1].weight.normal_(0, 0.05)
                    blk.norm_2.fc[-1].weight.normal_(0, 0.05)
        torch._dynamo.config.optimize_ddp = False
        mc = torch.compile(m, dynamic=True)
        run_two_steps(mc, f"A/外层 torch.compile + direct [{_mode}]")
    except Exception as e:
        check(f"A/外层 torch.compile + direct [{_mode}]", False,
              f"{type(e).__name__}: {str(e)[:300]}")
```

B（`enable_compile` + 保守力，make_fx）、C（编译 vs eager 数值一致，保守力）、
D（`enable_compile` + direct，plain_compile）同样加 `for _mode in ("input", "adanorm")`
外层循环，并把 `build(...)` 补上 `scd_inject=_mode`；C、D 段里成对构造的
eager/编译模型都要传同一个 `scd_inject=_mode`。

- [ ] **Step 2: 运行**

`RUN experimental/tests/test_equiformer_v3_scd.py`

Expected: `ALL PASS`。编译段从 8 项变为 16 项，总计 60 项。

若 B/D 段在 `adanorm` 下报 stale-bake 或形状错误，检查 Task 3 中
`_conservative_compiled_forward` 的 `force_args` / `force_dyn` / `_prime` 三处是否
都按 `cond is None` 分支构造 —— 漏掉 `_prime` 只会在 `compile_dynamic=True` 下暴露。

- [ ] **Step 3: Commit**

```bash
git add experimental/tests/test_equiformer_v3_scd.py
git commit -m "SCD v1: 编译路径 × adanorm 验证

四段编译测试（外层 torch.compile / make_fx 保守力 / 数值一致 / plain_compile）
各自对 input 与 adanorm 两种注入模式跑一遍，确认 cond 作为显式 traced 入参
不会 stale-bake，且编译-eager 数值一致。"
```

---

## 完成标准

- [ ] 三个测试文件全部 `ALL PASS`
- [ ] `scd_inject` 三种模式 × (direct, 保守力) 全部跑通
- [ ] identity-init 断言通过：未训练调制头时 adanorm 模型 == 同权重 DeNS
- [ ] adanorm 模式旋转不变/等变 `< 1e-4`
- [ ] 四档冻结的参数计数精确匹配
- [ ] gradient checkpointing 下 cond 正确透传
- [ ] 配置可实例化且 `use_compile: False`
- [ ] （Task 9）四段编译测试对 `input` / `adanorm` 两种模式均通过

## 已知遗留（不在本计划范围）

- **bf16 下 AdaNorm 的精度交互**：现有 norm 带 `@autocast(enabled=False)` 的 fp32
  守卫，而 AdaNorm 的调制 MLP 在守卫之外。`EquivariantAdaNorm.forward` 已用
  `.to(x.dtype)` 对齐 dtype，但 bf16 下是否引入新的 fp32 island 未测。
- **DropPath 0.1 的实际效果**：per-graph 语义下强度未知，列为第一待调项。
