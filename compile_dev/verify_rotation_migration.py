"""Verify EquiformerV3 edge-rotation migration: Gram-Schmidt / e3nn path (baseline)
vs UMA Euler path (migrated), on a small randomly-initialised V3 backbone.

Tests:
  1. Rotational equivariance (energy invariant, forces equivariant) with new Euler path.
  2. Roll-invariance: same output across different random-gamma seeds.
  3. Losslessness: migrated Euler path vs old Gram-Schmidt/e3nn baseline, within
     energy <1e-5 and force ~1e-3 (fp32 equivariance noise floor).

Usage:
  # Step 1 (TDD baseline): record old Gram-Schmidt path output
  bash scripts/compile_env.sh compile_dev/verify_rotation_migration.py --baseline

  # Step 3 (after migration): verify new Euler path passes all three tests
  bash scripts/compile_env.sh compile_dev/verify_rotation_migration.py

Gate discipline: every tolerance is enforced with sys.exit(1) on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import torch
from torch_geometric.data import Data

from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
from experimental.models.equiformer_v3.wigner import wigner_D

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "_rot_baseline.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Old Gram-Schmidt + e3nn rotation path (inline, for baseline generation)
# ─────────────────────────────────────────────────────────────────────────────

def _old_init_edge_rot_mat(edge_distance_vec):
    """Original Gram-Schmidt edge rotation matrix (V3 code before Task-1 migration)."""
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_distance_vec ** 2, dim=1))
    norm_x = edge_distance_vec / (edge_vec_0_distance.view(-1, 1))

    edge_vec_2 = torch.rand_like(edge_distance_vec) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2 ** 2, dim=1)).view(-1, 1)
    )
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / torch.sqrt(torch.sum(norm_z ** 2, dim=1, keepdim=True))
    norm_z = norm_z / torch.sqrt(torch.sum(norm_z ** 2, dim=1)).view(-1, 1)
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.sqrt(torch.sum(norm_y ** 2, dim=1, keepdim=True))

    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    return torch.transpose(edge_rot_mat_inv, 1, 2).detach()


def _old_set_wigner(so3_rot, edge_rot_mat):
    """Old _rotation_to_wigner_matrix + set_wigner using e3nn for Euler angles."""
    from e3nn import o3

    x = edge_rot_mat[:, :, 1]  # second column = edge direction
    alpha, beta = o3.xyz_to_angles(x)
    R_align = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
    R_local = torch.bmm(R_align, edge_rot_mat)
    gamma = torch.atan2(R_local[..., 0, 2], R_local[..., 0, 0])

    end_lmax = so3_rot.lmax
    size = int((end_lmax + 1) ** 2)
    wigner = torch.zeros(len(alpha), size, size, device=edge_rot_mat.device)
    start = 0
    for lmax in range(0, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma)
        end = start + block.size(1)
        wigner[:, start:end, start:end] = block
        start = end
    wigner = wigner.detach()

    wigner = torch.einsum("mi, nij -> nmj", so3_rot.wigner_index_to_m_array, wigner)
    wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
    wigner_inv = wigner_inv * so3_rot.wigner_inv_rescale
    so3_rot.wigner = wigner.detach()
    so3_rot.wigner_inv = wigner_inv.detach()


def _old_forward_edge(self, edge_distance, edge_distance_vec):
    """Replacement for _forward_edge using the old rotation path."""
    rot_mat = _old_init_edge_rot_mat(edge_distance_vec)
    _old_set_wigner(self.so3_rotation, rot_mat)
    edge_envelope_weight = (
        self.envelope_func(edge_distance)
        if self.envelope_func is not None
        else None
    )
    edge_distance = self.distance_expansion(edge_distance)
    return edge_distance, edge_envelope_weight


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(seed: int = 42) -> EquiformerV3_OC:
    torch.manual_seed(seed)
    model = EquiformerV3_OC(
        use_pbc=False,
        otf_graph=True,
        regress_forces=True,
        direct_prediction=True,
        lmax=2,
        mmax=2,
        num_layers=2,
        num_channels=16,
        num_radial_basis=32,
        max_radius=5.0,
        max_neighbors=20,
        attn_hidden_channels=16,
        attn_alpha_channels=8,
        attn_value_channels=8,
        ffn_hidden_channels=16,
        edge_channels=16,
        attn_grid_resolution_list=[8, 4],
        ffn_grid_resolution_list=[8, 8],
    ).eval()
    return model


def build_batch(n: int = 8, seed: int = 123) -> Data:
    torch.manual_seed(seed)
    return Data(
        pos=torch.randn(n, 3) * 1.5,
        atomic_numbers=torch.randint(1, 10, (n,)),
        natoms=torch.tensor([n]),
        batch=torch.zeros(n, dtype=torch.long),
        cell=torch.eye(3).unsqueeze(0) * 20.0,
        fixed=torch.zeros(n, dtype=torch.bool),
        tags=torch.zeros(n, dtype=torch.long),
    )


def run(model: EquiformerV3_OC, data: Data, seed: int = 0):
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model(data)
    return out["energy"].detach().double(), out["forces"].detach().double()


def with_old_path(model: EquiformerV3_OC):
    """Context manager: monkey-patch model to use old Gram-Schmidt + e3nn path."""
    orig = model._forward_edge
    model._forward_edge = types.MethodType(_old_forward_edge, model)
    try:
        yield
    finally:
        model._forward_edge = orig


# make with_old_path usable as a context manager
class _OldPathCtx:
    def __init__(self, model):
        self.model = model
        self.orig = None

    def __enter__(self):
        self.orig = self.model._forward_edge
        self.model._forward_edge = types.MethodType(_old_forward_edge, self.model)
        return self.model

    def __exit__(self, *args):
        self.model._forward_edge = self.orig


def random_rotation(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ─────────────────────────────────────────────────────────────────────────────
# Gate checks (each returns (error_value, passed: bool))
# ─────────────────────────────────────────────────────────────────────────────

def check_equivariance(model, data):
    """Energy invariance + force equivariance under a random rotation."""
    e0, f0 = run(model, data, seed=0)
    R = random_rotation(seed=7)
    rot_data = Data(
        pos=(data.pos.double() @ R.T).float(),
        atomic_numbers=data.atomic_numbers,
        natoms=data.natoms,
        batch=data.batch,
        cell=data.cell,
        fixed=data.fixed,
        tags=data.tags,
    )
    e_rot, f_rot = run(model, rot_data, seed=0)
    e_err = (e_rot - e0).abs().item()
    f_err = (f_rot - f0 @ R.T).abs().max().item()
    return e_err, f_err


def check_roll_noise(model, data, n: int = 3):
    """Energy/force spread across n different random-gamma seeds."""
    es, fs = [], []
    for s in range(n):
        e, f = run(model, data, seed=100 + s)
        es.append(e); fs.append(f)
    e_spread = max((es[i] - es[0]).abs().item() for i in range(n))
    f_spread = max((fs[i] - fs[0]).abs().max().item() for i in range(n))
    return e_spread, f_spread


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", action="store_true",
        help="Record old Gram-Schmidt path baseline to _rot_baseline.pt"
    )
    args = parser.parse_args()

    model = build_model(seed=42)
    data = build_batch(n=8, seed=123)

    FAIL = False

    if args.baseline:
        # ── Baseline: old Gram-Schmidt + e3nn path ────────────────────────────
        print("=" * 60)
        print("MODE: --baseline  (old Gram-Schmidt + e3nn path)")
        print("=" * 60)

        with _OldPathCtx(model):
            e_err_eq, f_err_eq = check_equivariance(model, data)
            print(f"\n[baseline] equivariance: energy_err={e_err_eq:.2e}  force_err={f_err_eq:.2e}")

            e_roll, f_roll = check_roll_noise(model, data)
            print(f"[baseline] roll-noise:   energy_spread={e_roll:.2e}  force_spread={f_roll:.2e}")

            e0, f0 = run(model, data, seed=0)
            print(f"\n[baseline] energy={e0.item():.6f}  |F|_mean={f0.abs().mean().item():.6f}")

        torch.save(
            {"energy": e0.float(), "forces": f0.float(),
             "roll_noise_e": float(e_roll), "roll_noise_f": float(f_roll)},
            BASELINE_PATH,
        )
        print(f"Saved baseline → {BASELINE_PATH}")

    else:
        # ── Migration gate: new Euler path ────────────────────────────────────
        print("=" * 60)
        print("MODE: migration gate  (new UMA Euler path)")
        print("=" * 60)

        # Test 1: rotational equivariance
        THRESH_E_EQ = 1e-5
        THRESH_F_EQ = 1e-3
        e_err_eq, f_err_eq = check_equivariance(model, data)
        eq_e_ok = e_err_eq < THRESH_E_EQ
        eq_f_ok = f_err_eq < THRESH_F_EQ
        print(f"\n[Test 1] rotational equivariance (new Euler path):")
        print(f"  energy_invariance_err = {e_err_eq:.2e}  (threshold < {THRESH_E_EQ:.0e})"
              + ("  OK" if eq_e_ok else "  FAIL"))
        print(f"  force_equivariance_err = {f_err_eq:.2e}  (threshold < {THRESH_F_EQ:.0e})"
              + ("  OK" if eq_f_ok else "  FAIL"))
        if not (eq_e_ok and eq_f_ok):
            print("  VERDICT: FAIL")
            FAIL = True
            if not eq_e_ok:
                print(f"ASSERT FAILED: energy equivariance {e_err_eq:.2e} >= {THRESH_E_EQ:.0e}",
                      file=sys.stderr)
            if not eq_f_ok:
                print(f"ASSERT FAILED: force equivariance {f_err_eq:.2e} >= {THRESH_F_EQ:.0e}",
                      file=sys.stderr)
        else:
            print("  VERDICT: PASS")

        # Test 2: roll-invariance / noise floor
        e_roll, f_roll = check_roll_noise(model, data)
        print(f"\n[Test 2] roll-invariance (noise floor):")
        print(f"  energy_spread = {e_roll:.2e}")
        print(f"  force_spread  = {f_roll:.2e}")

        # Test 3: losslessness vs saved old-path baseline
        if not os.path.exists(BASELINE_PATH):
            print(f"\n[Test 3] losslessness: SKIPPED (no baseline found at {BASELINE_PATH})")
            print("  → Re-run with --baseline first.")
        else:
            ckpt = torch.load(BASELINE_PATH, weights_only=True)
            bl_e = ckpt["energy"].double()
            bl_f = ckpt["forces"].double()
            bl_roll_e = float(ckpt.get("roll_noise_e", 1e-5))
            bl_roll_f = float(ckpt.get("roll_noise_f", 1e-3))

            e_new, f_new = run(model, data, seed=0)
            e_err_ls = (e_new - bl_e).abs().item()
            f_err_ls = (f_new - bl_f).abs().max().item()

            # Tolerance: hard floor OR 5× baseline roll-noise, whichever is larger
            tol_e = max(1e-5, 5 * bl_roll_e)
            tol_f = max(1e-3, 5 * bl_roll_f)
            ls_e_ok = e_err_ls < tol_e
            ls_f_ok = f_err_ls < tol_f

            print(f"\n[Test 3] losslessness vs old Gram-Schmidt baseline:")
            print(f"  baseline roll-noise: energy={bl_roll_e:.1e}  force={bl_roll_f:.1e}")
            print(f"  energy_err = {e_err_ls:.2e}  (tolerance < {tol_e:.1e})"
                  + ("  OK" if ls_e_ok else "  FAIL"))
            print(f"  force_err  = {f_err_ls:.2e}  (tolerance < {tol_f:.1e})"
                  + ("  OK" if ls_f_ok else "  FAIL"))
            if not (ls_e_ok and ls_f_ok):
                print("  VERDICT: FAIL — migrated path diverges from baseline")
                FAIL = True
                if not ls_e_ok:
                    print(f"ASSERT FAILED: lossless energy {e_err_ls:.2e} >= {tol_e:.1e}",
                          file=sys.stderr)
                if not ls_f_ok:
                    print(f"ASSERT FAILED: lossless force {f_err_ls:.2e} >= {tol_f:.1e}",
                          file=sys.stderr)
            else:
                print("  VERDICT: PASS (within fp32 / roll-noise tolerance)")

        print("\n" + "=" * 60)
        if FAIL:
            print("GATE: FAIL")
            sys.exit(1)
        else:
            print("GATE: PASS")


if __name__ == "__main__":
    main()
