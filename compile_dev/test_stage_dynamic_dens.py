"""Dynamic=True gate (DeNS): single-kernel assertion for the conservative-force
compile path with compile_dynamic=True on the DeNS backbone.

Same single-kernel assertion as test_stage_dynamic_base.py but on EquiformerV3DeNS_OC
with regress_stress=True and a denoising-active batch — validates the stress+DeNS
core_fn (force_embedding explicit input + double backward wrt pos AND disp) under
symbolic dynamic trace.

Verifies:
  (1) CompiledForceRegion cache stays size 1 across 3 distinct shapes (ZERO recompile).
  (2) Forces / stress finite and numerically consistent with eager (<1e-5; probe saw ~4e-9).
  (3) Shapes 2+ run in <2s (zero recompile proxy).

Requires torch>=2.11 for dynamic symbolic make_fx. sys.exit(1) on any failure.

Run:
    bash scripts/compile_env.sh compile_dev/test_stage_dynamic_dens.py
"""

from __future__ import annotations

import sys
import time

import torch
from torch_geometric.data import Data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def _patch_deterministic_rotation():
    import torch.nn.functional as F
    import experimental.models.equiformer_v3.equiformer_v3 as ev3
    from experimental.models.equiformer_v3.edge_rot_mat import Safeacos, Safeatan2

    def _det_eulers(edge_distance_vec):
        xyz = F.normalize(edge_distance_vec).clamp(-1.0, 1.0)
        x, y, z = torch.split(xyz, 1, dim=1)
        beta = Safeacos.apply(y.squeeze(-1))
        alpha = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))
        gamma = torch.zeros_like(alpha)
        return -gamma, -beta, -alpha

    ev3.init_edge_rot_euler_angles = _det_eulers


_patch_deterministic_rotation()

_CFG = dict(
    use_pbc=False,
    otf_graph=True,
    regress_forces=True,
    regress_stress=True,
    direct_prediction=False,
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
    attn_weights_drop=0.0,
    drop_path_rate=0.0,
    alpha_drop=0.0,
    value_drop=0.0,
    proj_drop=0.0,
    ffn_drop=0.0,
    attn_mask_rate=0.0,
    enable_compile=True,
    compile_dynamic=True,
)


def build_model(seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC
    torch.manual_seed(seed)
    return EquiformerV3DeNS_OC(**_CFG).to(DEVICE).to(DTYPE).train()


def build_batch(natoms_list, seed: int = 0):
    torch.manual_seed(seed)
    N = int(sum(natoms_list))
    pos = torch.rand(N, 3) * 3.0
    an = torch.randint(1, 10, (N,))
    batch = torch.cat(
        [torch.full((n,), i) for i, n in enumerate(natoms_list)]
    ).long()
    cell = torch.stack([torch.eye(3) * 10.0 for _ in natoms_list])
    data = Data(
        pos=pos,
        atomic_numbers=an,
        natoms=torch.tensor(natoms_list),
        batch=batch,
        cell=cell,
        fixed=torch.zeros(N, dtype=torch.bool),
        tags=torch.zeros(N, dtype=torch.long),
    )
    # DeNS: denoising-active batch with mixed noise mask and stress live
    data.denoising_pos_forward = True
    data.forces = torch.randn(N, 3)
    nm = torch.zeros(N, dtype=torch.bool)
    nm[::2] = True
    data.noise_mask = nm
    data.dens_batch_mask = torch.zeros(len(natoms_list), dtype=torch.bool)
    data = data.to(DEVICE)
    data._orig_pos = data.pos.detach().clone()
    data._orig_cell = data.cell.detach().clone()
    return data


def reset_data(data):
    data.pos = data._orig_pos.detach().clone()
    data.cell = data._orig_cell.detach().clone()
    return data


def run_forward(model, data, compiled: bool, seed: int = 0):
    reset_data(data)
    torch.manual_seed(seed)
    model.enable_compile = compiled
    return model._forward_gradient(data)


def main():
    if torch.__version__ < "2.11":
        print(f"SKIP: dynamic=True requires torch>=2.11 (have {torch.__version__})")
        sys.exit(0)

    print("=" * 70)
    print(f"Dynamic=True gate [DeNS]  device={DEVICE}  torch={torch.__version__}")
    print("=" * 70)

    # 3 distinct system sizes: different natoms AND nedges
    shapes = [
        ([5, 7], 11),    # 12 atoms
        ([6, 4, 3], 22), # 13 atoms
        ([8, 5], 33),    # 13 atoms, different nedges (different seed)
    ]

    model = build_model(seed=42)
    ok = True
    cache_sizes = []
    compile_time = None

    for idx, (natoms_list, seed) in enumerate(shapes):
        data = build_batch(natoms_list, seed=seed)
        natoms = int(sum(natoms_list))
        nsys = len(natoms_list)

        # Eager reference
        out_e = run_forward(model, data, compiled=False, seed=0)
        f_e = out_e.get("forces")
        s_e = out_e.get("stress")

        # Compiled run
        reset_data(data)
        torch.manual_seed(0)
        model.enable_compile = True
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out_c = model._forward_gradient(data)
        except Exception as e:
            print(f"  shape{idx+1} FAIL (compiled raised): {type(e).__name__}: {str(e)[:300]}")
            ok = False
            cache_sizes.append(None)
            continue
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_call = time.perf_counter() - t0

        if idx == 0:
            compile_time = t_call

        cache_size = len(model._compiled_region._cache) if model._compiled_region else 0
        cache_sizes.append(cache_size)

        f_c = out_c.get("forces")
        s_c = out_c.get("stress")
        ferr = (f_e - f_c.detach()).abs().max().item() if f_e is not None and f_c is not None else float("nan")
        serr = (s_e - s_c.detach()).abs().max().item() if s_e is not None and s_c is not None else float("nan")
        finite = (
            (torch.isfinite(f_c).all().item() if f_c is not None else False) and
            (torch.isfinite(s_c).all().item() if s_c is not None else False)
        )
        # shapes 2+ must be fast (no recompile); first compile is expected ~120s
        recompiled = (idx > 0) and (t_call > 2.0)

        shape_ok = cache_size == 1 and ferr < 1e-5 and serr < 1e-5 and finite and not recompiled
        ok = ok and shape_ok

        print(
            f"  shape{idx+1}: natoms={natoms} nsys={nsys} "
            f"cache={cache_size} t={t_call:.2f}s "
            f"f_err={ferr:.2e} s_err={serr:.2e} finite={finite} "
            f"{'RECOMPILE' if recompiled else ''} "
            f"-> {'OK' if shape_ok else 'FAIL'}"
        )

    # Assert 1: cache stays at 1 (single kernel, zero recompile via make_fx)
    single_kernel = all(c == 1 for c in cache_sizes if c is not None)
    print(f"\n  cache_sizes across shapes: {cache_sizes}  single_kernel={single_kernel}")
    if not single_kernel:
        print(f"  ASSERT FAILED: cache grew beyond 1 — dynamic=True not effective")
        ok = False
    if None in cache_sizes:
        print(f"  ASSERT FAILED: one or more shapes crashed — compile not working")
        ok = False

    print(f"\n{'GATE: PASS' if ok else 'GATE: FAIL'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
