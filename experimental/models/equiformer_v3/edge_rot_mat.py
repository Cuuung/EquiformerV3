import torch
import torch.nn.functional as F

from .wigner import wigner_D

# NOTE (rotation migration, Task 1):
# Migrated from the Gram-Schmidt + wigner-from-3x3-mat path to the UMA/esen
# Euler-angle path (fairchem 2.x esen/common/rotation.py). Blockers removed:
#   - edge_rot_mat.py:17-20  min-distance debug branch (data-dependent → breaks make_fx)
#   - edge_rot_mat.py:65     assert max(vec_dot) < 0.99  (data-dependent → breaks make_fx)
#   - so3.py  rot_clip boolean-mask (data-dependent index, use_rotation_mask guard)
# Gradient stability now comes from the clamped-safe backward of Safeacos/Safeatan2.
# Verified lossless on a small randomly-initialized V3 backbone (Task-1 gate).

# Kept for backward compatibility: so3.py imports this constant at module level.
_ROTATION_MASK_THRESHOLD = 0.999999

EPS = 1e-7


class Safeacos(torch.autograd.Function):
    """acos with a clamped-safe backward (avoids NaN gradients at |x|->1)."""

    @staticmethod
    def forward(ctx, x):
        x_clamped = x.clamp(-1 + EPS, 1 - EPS)
        ctx.save_for_backward(x_clamped)
        return torch.acos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x_clamped,) = ctx.saved_tensors
        denom = torch.sqrt(1 - x_clamped.pow(2)).clamp(min=EPS)
        return -grad_output / denom


class Safeatan2(torch.autograd.Function):
    """atan2 with a clamped-safe backward (avoids NaN gradients at the origin)."""

    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        denom = (x.pow(2) + y.pow(2)).clamp(min=EPS)
        return (x / denom) * grad_output, (-y / denom) * grad_output


def init_edge_rot_euler_angles(edge_distance_vec):
    """Edge direction -> intrinsic Euler angles (alpha, beta, gamma) aligning the
    edge to +Y, with a random roll (gamma) for SO(2) equivariance during training.

    make_fx-friendly: F.normalize (eps-safe) + clamp + Safeacos/Safeatan2; no
    data-dependent branches, no rot_clip boolean-mask.
    Migrated from fairchem 2.x esen/common/rotation.py::init_edge_rot_euler_angles.
    """
    # clamp because under compile, normalize can return >1.0 (pytorch #163082)
    xyz = F.normalize(edge_distance_vec).clamp(-1.0, 1.0)
    x, y, z = torch.split(xyz, 1, dim=1)
    beta = Safeacos.apply(y.squeeze(-1))   # polar angle from Y axis
    alpha = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))  # azimuthal in XZ plane
    gamma = torch.rand_like(alpha) * 2 * torch.pi  # random roll
    # intrinsic -> extrinsic swap
    return -gamma, -beta, -alpha


def eulers_to_wigner(eulers, start_lmax, end_lmax):
    """Build the block-diagonal Wigner-D matrix from Euler angles.

    Uses V3's wigner_D (global Jd, no Jd parameter).

    make_fx-friendly: static shapes, no boolean-mask / rot_clip. Gradient
    stability comes from the clamped-safe backward of Safeacos/Safeatan2.
    Migrated from fairchem 2.x esen/common/rotation.py::eulers_to_wigner
    (adapted to V3's wigner_D signature).
    """
    alpha, beta, gamma = eulers

    size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
    # Functional block-diagonal assembly (symbolic / dynamic-shape safe): pad each
    # Wigner block to (N, size, size) at its diagonal offset and sum. Avoids
    # torch.zeros(len(alpha), ...) (Python len() forces symbolic edge dim to a
    # constant) + in-place slice writes that break dynamic=True.
    wigner = None
    start = 0
    for lmax in range(start_lmax, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma)  # (N, dl, dl)
        dl = block.size(-1)
        end = start + dl
        # pad last two dims: (left, right, top, bottom)
        padded = F.pad(block, (start, size - end, start, size - end))
        wigner = padded if wigner is None else wigner + padded
        start = end

    return wigner
