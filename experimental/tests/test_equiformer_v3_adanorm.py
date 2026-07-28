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

# gate 契约：与 x 广播兼容；shared 下有意不展开
for scope in ("per_degree", "shared", "l0_only"):
    m = make(scope)
    x, cond = rand_inputs()
    out, gate = m(x, cond)
    check(f"gate 与 x 广播兼容 scope={scope}",
          (out * gate).shape == out.shape, f"gate={tuple(gate.shape)}")

m = make("shared")
x, cond = rand_inputs()
check("shared 下 gate 有意保持 [N, 1, C] 不展开",
      m(x, cond)[1].shape == (N, 1, C))

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
