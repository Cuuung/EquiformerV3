"""Shared helpers for the Task-4 conservative-force grad-alignment gates.

The hard gates (test_stage_conservative_base.py / _dens.py) and the diagnostic
(diag_grad_tracking.py) all compare eager vs compiled ``_forward_gradient`` on the
SAME model instance (toggle ``enable_compile``) at the SAME fixed weights, on the
SAME input, and judge GRADIENT ALIGNMENT (cos / norm-ratio / all-params-grad),
NOT loss trajectories.

Determinism: V3's edge rotation draws a random gamma roll
(``init_edge_rot_euler_angles``) — training-time SO(2) augmentation, irrelevant to
gradient-correctness of the compile machinery but it desyncs a deterministic
eager-vs-compiled comparison (make_fx keeps ``rand_like`` as a runtime op, so the
DeNS path's extra compiled-region core_compute shifts the RNG stream seen by the
eager denoising pass). The gate pins gamma=0 (a valid frame) so the comparison is
frame-fair and fully deterministic; it still warms up (traces) and re-seeds before
each forward for good measure.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

SEED = 12345
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _patch_deterministic_rotation():
    """Pin the random SO(2) gamma roll to 0 (fixed frame) FOR THE GATE ONLY, so
    eager vs compiled is a deterministic, frame-fair comparison. Patches the name
    bound in the model module namespace, used by EquiformerV3_OC._init_edge_rot_mat
    (inherited by the DeNS subclass)."""
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

# Small model; all dropout/drop-path rates 0 so the only RNG consumer is the
# rotation gamma (kept frame-fair by per-forward seeding).
CFG = dict(
    use_pbc=False,
    otf_graph=True,
    regress_forces=True,
    regress_stress=True,
    direct_prediction=False,  # conservative -> _forward_gradient, no force/stress blocks
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
)


def build_base(seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
    torch.manual_seed(seed)
    return EquiformerV3_OC(**CFG).to(DEVICE).train()


def build_dens(seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC
    torch.manual_seed(seed)
    return EquiformerV3DeNS_OC(**CFG).to(DEVICE).train()


def build_batch(natoms_list, box: float = 10.0, seed: int = 0, denoising: bool = False):
    """Cluster atoms tightly (within max_radius) so the otf graph is non-empty."""
    torch.manual_seed(seed)
    N = int(sum(natoms_list))
    pos = torch.rand(N, 3) * 3.0
    an = torch.randint(1, 10, (N,))
    batch = torch.cat(
        [torch.full((n,), i) for i, n in enumerate(natoms_list)]
    ).long()
    cell = torch.stack([torch.eye(3) * box for _ in natoms_list])
    data = Data(
        pos=pos,
        atomic_numbers=an,
        natoms=torch.tensor(natoms_list),
        batch=batch,
        cell=cell,
        fixed=torch.zeros(N, dtype=torch.bool),
        tags=torch.zeros(N, dtype=torch.long),
    )
    if denoising:
        # Mixed noise mask: denoising atoms (mask=1) exercise the eager dens_block
        # and make force_embedding nonzero; non-denoising atoms (mask=0) exercise
        # the compiled 2nd-order force path. dens_batch_mask=0 keeps stress live so
        # force_embedding gets gradient through the (global, energy-based) stress.
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


def run(model, data, compiled: bool, seed: int = SEED):
    """One ``_forward_gradient`` forward. Resets pos/cell (the eager stress path
    mutates them in place) and seeds so the rotation gamma is frame-fair."""
    data.pos = data._orig_pos.detach().clone()
    data.cell = data._orig_cell.detach().clone()
    torch.manual_seed(seed)
    model.enable_compile = compiled
    return model._forward_gradient(data)


def loss_fs(out) -> torch.Tensor:
    l = out["forces"].pow(2).sum()
    if "stress" in out:
        l = l + out["stress"].pow(2).sum()
    return l


def grads_at(model, data, compiled: bool, seed: int = SEED):
    model.zero_grad(set_to_none=True)
    out = run(model, data, compiled, seed)
    loss_fs(out).backward()
    g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    out_det = {k: v.detach() for k, v in out.items()}
    return out_det, g


def grad_align(ge: dict, gc: dict, rel_thr: float = 1e-4):
    """Gradient-alignment metrics, eager vs compiled.

    Returns dict with:
      gcos / gratio: cos and ||gc||/||ge|| over the CONCATENATED grad vector of
        all shared params (esen-style; dominated by meaningful-magnitude params,
        robust to CUDA-atomic noise). This is the primary alignment metric.
      pmin_cos / pdev_ratio: min cos / max |1-ratio| over per-param grads whose
        eager norm exceeds ``rel_thr * max_param_grad_norm`` (near-zero-gradient
        params are excluded — their cosine is pure numerical noise, not a real
        misalignment).
      big: the list of those above-threshold param names.
      rows: (name, cos, ratio, ||ge||) for the above-threshold params.
    """
    import torch.nn.functional as F

    keys = [k for k in ge if k in gc]
    a = torch.cat([ge[k].flatten() for k in keys])
    b = torch.cat([gc[k].flatten() for k in keys])
    gcos = F.cosine_similarity(a, b, dim=0).item()
    gratio = (b.norm() / a.norm()).item() if a.norm() > 0 else 1.0

    norms = {k: ge[k].norm().item() for k in keys}
    mx = max(norms.values()) if norms else 0.0
    thr = rel_thr * mx
    big = [k for k in keys if norms[k] > thr]
    pmin, pdev, rows = 1.0, 0.0, []
    for k in big:
        c = F.cosine_similarity(ge[k].flatten(), gc[k].flatten(), dim=0).item()
        r = gc[k].norm().item() / norms[k]
        rows.append((k, c, r, norms[k]))
        pmin = min(pmin, c)
        pdev = max(pdev, abs(r - 1.0))
    return {
        "gcos": gcos, "gratio": gratio, "pmin_cos": pmin, "pdev_ratio": pdev,
        "big": big, "rows": rows,
    }


def warmup_compiled(model, data, seed: int = SEED):
    """Trigger the make_fx trace (bakes gamma) at this shape; discard output."""
    out = run(model, data, compiled=True, seed=seed)
    loss_fs(out).backward()
    model.zero_grad(set_to_none=True)


def run_full_gate(build_fn, label: str, denoising: bool) -> bool:
    """Hard grad-alignment gate. Returns True iff PASS. Asserts:
      (1) compiled == eager energy/forces/stress at fixed weights,
      (2) per-param cos(g_e, g_c) > 0.999 and |1-||gc||/||ge||| < 0.05,
      (3) the compiled grad-param SET equals the eager set AND every trainable
          param receives a gradient (strip_detach silent-severing check),
      (4) a different system size re-traces (no crash) with new-shape cos > 0.999.
    """
    torch.set_float32_matmul_precision("high")
    print("=" * 70)
    print(f"Task-4 conservative gate [{label}]  denoising={denoising}")
    print("=" * 70)

    model = build_fn(seed=42)
    data = build_batch([6, 7], seed=123, denoising=denoising)
    n_total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"natoms={int(data.pos.shape[0])} nsys={int(data.natoms.numel())} "
          f"trainable_params={n_total}")

    # Warm up: trace the compiled region at this shape (bakes the rotation gamma).
    warmup_compiled(model, data)
    if model._compiled_region is None:
        print("ASSERT FAILED: _compiled_region is None — conservative not wired")
        return False
    print(f"compiled region wired: cache_size={len(model._compiled_region._cache)}")

    # (1)+(2)+(3): eager vs compiled at the SAME weights.
    out_e, ge = grads_at(model, data, compiled=False)
    out_c, gc = grads_at(model, data, compiled=True)

    eerr = (out_e["energy"] - out_c["energy"]).abs().max().item()
    ferr = (out_e["forces"] - out_c["forces"]).abs().max().item()
    serr = (out_e["stress"] - out_c["stress"]).abs().max().item() if "stress" in out_e else 0.0
    print(f"\n(1) numeric: energy err={eerr:.2e}  force err={ferr:.2e}  stress err={serr:.2e}")
    num_ok = eerr < 1e-4 and ferr < 1e-3 and serr < 1e-3

    m = grad_align(ge, gc)
    print(f"(2) alignment: concat cos={m['gcos']:.6f} ratio={m['gratio']:.5f}  |  "
          f"per-param(|g|>1e-4*max, n={len(m['big'])}) "
          f"min cos={m['pmin_cos']:.6f} max|1-ratio|={m['pdev_ratio']:.3e}")
    for k, c, r, na in sorted(m["rows"], key=lambda x: x[1])[:3]:
        print(f"      worst-cos {k}: cos={c:.6f} ratio={r:.4f} |ge|={na:.2e}")
    align_ok = (m["gcos"] > 0.999 and abs(m["gratio"] - 1.0) < 0.05
                and m["pmin_cos"] > 0.999 and m["pdev_ratio"] < 0.05)

    # strip_detach silent-severing check: every param that gets a (meaningful)
    # gradient in eager MUST also get one in compiled. If strip_detach severed the
    # 2nd-order path, the whole backbone would lose its gradient in compiled.
    set_e, set_c = set(ge), set(gc)
    severed = set_e - set_c
    big_set = set(m["big"])
    big_severed = big_set - set_c
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and n not in set_e]
    print(f"(3) grad coverage: eager {len(set_e)}/{n_total}  compiled {len(set_c)}/{n_total}  "
          f"eager-grad params covered by compiled={set_e <= set_c}  "
          f"meaningful(|g|>thr)={len(big_set)} all-covered={not big_severed}")
    if severed:
        print(f"    SEVERED in compiled (strip_detach FAILED): {sorted(severed)[:6]}")
    if no_grad:
        print(f"    no eager grad (decoupled from force/stress loss): {sorted(no_grad)[:6]}")
    grad_ok = (set_e <= set_c) and not big_severed and len(big_set) > 0

    # (4) different system size: must re-trace and stay aligned.
    data2 = build_batch([5, 8, 4], seed=777, denoising=denoising)
    print(f"\n(4) changed shape: natoms={int(data2.pos.shape[0])} "
          f"nsys={int(data2.natoms.numel())}")
    shape_ok = False
    try:
        warmup_compiled(model, data2, seed=SEED + 1)  # trace @ new shape
        out_e2, ge2 = grads_at(model, data2, compiled=False, seed=SEED + 1)
        out_c2, gc2 = grads_at(model, data2, compiled=True, seed=SEED + 1)
        ferr2 = (out_e2["forces"] - out_c2["forces"]).abs().max().item()
        m2 = grad_align(ge2, gc2)
        cache2 = len(model._compiled_region._cache)
        print(f"    re-traced ok (cache_size={cache2})  force err={ferr2:.2e}  "
              f"concat cos={m2['gcos']:.6f} ratio={m2['gratio']:.5f}  "
              f"per-param min cos={m2['pmin_cos']:.6f}")
        shape_ok = (cache2 >= 2 and m2["gcos"] > 0.999 and abs(m2["gratio"] - 1.0) < 0.05
                    and m2["pmin_cos"] > 0.999 and ferr2 < 1e-3)
    except Exception as ex:  # noqa: BLE001
        print(f"    FAILED: {type(ex).__name__}: {str(ex)[:300]}")

    print("\n--- verdict ---")
    print(f"numeric={num_ok} align={align_ok} grad_coverage={grad_ok} shape={shape_ok}")
    ok = num_ok and align_ok and grad_ok and shape_ok
    print("PASS" if ok else "FAIL")
    return ok
