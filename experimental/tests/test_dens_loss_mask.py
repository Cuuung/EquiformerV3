"""`EquiformerV3DeNSTrainer._compute_loss` 的 mask 形状回归测试。

背景：`all_atoms=True` 且 batch 没有 `noise_mask` 时（即纯 SCD 预训练：全原子
加噪、`corrupt_ratio=None` 所以不做部分腐蚀），loss 路径里的 mask 会被拿去对
形状 [N, 3] 的力张量做**布尔索引**，因此必须保持 [N]。曾误写成 `.view(-1, 1)`，
在该组合下抛
    IndexError: The shape of the mask [N, 1] at index 1 does not match
                the shape of the indexed tensor [N, 3] at index 1
而所有既有配置都设了 `corrupt_ratio`，走的是 hybrid-loss 那一支（乘法，需要
[N, 1]），所以这条路径此前从未被覆盖。

本测试用最小 stub 绑定真实的 `_compute_loss`，不构造完整 trainer（后者要加载
数据集与优化器）。
"""
import sys
import types

import torch
from torch_geometric.data import Batch, Data

from fairchem.core.common.utils import setup_imports

setup_imports()

from fairchem.experimental.trainers.equiformer_v3_dens_trainer import (
    DenoisingPosParams,
    EquiformerV3DeNSTrainer,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILURES.append(name)


class _Normalizer:
    """恒等 normalizer，只需支持 .norm()。"""

    def norm(self, t):
        return t


def make_stub(all_atoms):
    stub = types.SimpleNamespace()
    stub.loss_functions = [
        ("forces", {"fn": lambda pred, target, natoms=None: (pred - target).abs().mean(),
                    "coefficient": 10.0})
    ]
    stub.output_targets = {
        "forces": {"level": "atom", "train_on_free_atoms": True}
    }
    stub.normalizers = {"denoising_pos_target": _Normalizer()}
    stub.denoising_pos_params = DenoisingPosParams(
        all_atoms=all_atoms,
        denoising_pos_coefficient=1.0,
        coefficient_linear_decay_min_factor=None,
    )
    stub._compute_current_denoising_pos_coefficient = types.MethodType(
        EquiformerV3DeNSTrainer._compute_current_denoising_pos_coefficient, stub
    )
    stub._compute_loss = types.MethodType(EquiformerV3DeNSTrainer._compute_loss, stub)
    return stub


def make_batch(n=5, with_noise_mask=False):
    d = Data(
        pos=torch.zeros(n, 3),
        natoms=torch.tensor([n]),
        fixed=torch.zeros(n),
        forces=torch.randn(n, 3),
        noise_vec=torch.randn(n, 3) * 0.04,
        energy=torch.tensor([0.0]),
    )
    b = Batch.from_data_list([d]).to(DEV)
    b.denoising_pos_forward = True
    if with_noise_mask:
        b.noise_mask = torch.ones(n, dtype=torch.bool, device=DEV)
    return b


# 触发过 IndexError 的那个组合：all_atoms=True 且无 noise_mask
try:
    b = make_batch(with_noise_mask=False)
    out = {"forces": torch.randn_like(b.forces)}
    loss = make_stub(all_atoms=True)._compute_loss(out, b)
    check("all_atoms=True 且无 noise_mask（纯 SCD：全原子加噪）",
          torch.is_tensor(loss) or isinstance(loss, (int, float)),
          f"loss={float(loss):.4f}")
except Exception as e:
    check("all_atoms=True 且无 noise_mask（纯 SCD：全原子加噪）", False, repr(e))

# 既有配置走的那一支：有 noise_mask → hybrid loss（乘法，需要 [N,1]）
try:
    b = make_batch(with_noise_mask=True)
    out = {"forces": torch.randn_like(b.forces)}
    loss = make_stub(all_atoms=True)._compute_loss(out, b)
    check("all_atoms=True 且有 noise_mask（DeNS 部分腐蚀，既有路径未回归）",
          torch.is_tensor(loss), f"loss={float(loss):.4f}")
except Exception as e:
    check("all_atoms=True 且有 noise_mask（DeNS 部分腐蚀，既有路径未回归）", False, repr(e))

# all_atoms=False：mask 来自 batch.fixed，不进上面那段分支
try:
    b = make_batch(with_noise_mask=False)
    out = {"forces": torch.randn_like(b.forces)}
    loss = make_stub(all_atoms=False)._compute_loss(out, b)
    check("all_atoms=False（DeNS 默认，只训自由原子）",
          torch.is_tensor(loss), f"loss={float(loss):.4f}")
except Exception as e:
    check("all_atoms=False（DeNS 默认，只训自由原子）", False, repr(e))

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL PASS")
