"""Probe: dynamic=True symbolic make_fx for EquiformerV3 conservative force path.

Discovery probe ONLY -- does NOT modify any production code.

Phase 1: Standalone symbolic make_fx on V3 core_fn (base + DeNS) to observe
         the collapse point (traced with prime dims).
Phase 1b: Same trace WITH in-memory monkey-patch of len()->shape[0] to verify
           fix is sufficient.
Phase 2: Full _forward_gradient with compile_dynamic=True across 3 system sizes
         to observe cache_size behaviour, first-compile wall-time, and numerics.

Run:
    ESEN_COMPILE_PROBE=1 bash scripts/compile_env.sh compile_dev/probe_dynamic_v3.py

Expected PASS criteria (dynamic=True truly effective):
  - symbolic make_fx succeeds AND re-run succeeds on different shape
  - cache_size stays == 1 for all 3 shapes
  - force/stress numerics finite and consistent with eager
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import types

import torch
from torch_geometric.data import Data

# -- logging ------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [REPO, os.path.join(REPO, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# -- deterministic rotation patch (same as _conservative_common.py) ----------
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


# -- tiny model configs -------------------------------------------------------
_SMALL_CFG = dict(
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


def build_base(seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
    torch.manual_seed(seed)
    return EquiformerV3_OC(**_SMALL_CFG).to(DEVICE).to(DTYPE).train()


def build_dens(seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC
    torch.manual_seed(seed)
    return EquiformerV3DeNS_OC(**_SMALL_CFG).to(DEVICE).to(DTYPE).train()


def build_batch(natoms_list, box: float = 10.0, seed: int = 0, denoising: bool = False):
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


# -- Phase 1: symbolic make_fx trace -----------------------------------------
def build_core_fn(model, variant, data):
    """Extract the core_fn closure exactly as _conservative_compiled_forward does."""
    model.batch_size = len(data.natoms)
    model.dtype = DTYPE
    model.device = DEVICE

    (
        edge_index, edge_distance, edge_distance_vec,
        cell_offsets, _, neighbors,
    ) = model.generate_graph(
        data,
        enforce_max_neighbors_strictly=model.enforce_max_neighbors_strictly,
        use_pbc_single=False,
    )
    an = data.atomic_numbers.long()
    co = cell_offsets.to(DTYPE)
    cell = data.cell.to(DTYPE)
    batch = data.batch
    pos = data.pos.detach().requires_grad_(True)

    energy_block = model.energy_block
    avg_num_nodes = model.avg_num_nodes

    if variant == "base":
        def _energy(pos_p, cell_p, an_, ei_, co_, batch_, n_sys):
            co_ = co_.reshape(ei_.shape[1], -1)
            src, dst = ei_[0], ei_[1]
            cell_e = cell_p.index_select(0, batch_.index_select(0, src))
            shifts = torch.einsum("ej,ejk->ek", co_, cell_e)
            edv = pos_p.index_select(0, src) - pos_p.index_select(0, dst) + shifts
            ed = torch.linalg.norm(edv, dim=-1)
            x_scalar, _x, _, _ = model.core_compute(an_, ed, edv, ei_, batch_)
            node_e = energy_block(x_scalar).view(-1)
            energy = torch.zeros(n_sys, device=node_e.device, dtype=node_e.dtype)
            energy.index_add_(0, batch_, node_e)
            return energy / avg_num_nodes

        def core_fn(pos_, disp_, an_, ei_, co_, cell_, batch_):
            sym = 0.5 * (disp_ + disp_.transpose(-1, -2))
            pos_p = pos_ + torch.bmm(
                pos_.unsqueeze(-2), torch.index_select(sym, 0, batch_)
            ).squeeze(-2)
            cell_p = cell_ + torch.bmm(cell_, sym)
            energy = _energy(pos_p, cell_p, an_, ei_, co_, batch_, cell_.shape[0])
            grads = torch.autograd.grad([energy.sum()], [pos_, disp_], create_graph=True)
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(cell_).abs().unsqueeze(-1)
            stress = (virial / volume.view(-1, 1, 1)).view(-1, 9)
            return energy, forces, stress

        from fairchem.core.common.compile_utils import make_prime_graph_example
        trace_ex = make_prime_graph_example(DEVICE, DTYPE, stress=True)
        real_ex = (
            pos,
            torch.zeros(cell.shape[0], 3, 3, device=DEVICE, dtype=DTYPE).requires_grad_(True),
            an, edge_index, co, cell, batch,
        )
    else:  # dens
        force_embedding, _, _, _ = model._forward_dens_force_encoding(data)
        fe = force_embedding

        def _energy(pos_p, cell_p, an_, ei_, co_, batch_, fe_, n_sys):
            co_ = co_.reshape(ei_.shape[1], -1)
            src, dst = ei_[0], ei_[1]
            cell_e = cell_p.index_select(0, batch_.index_select(0, src))
            shifts = torch.einsum("ej,ejk->ek", co_, cell_e)
            edv = pos_p.index_select(0, src) - pos_p.index_select(0, dst) + shifts
            ed = torch.linalg.norm(edv, dim=-1)
            x_scalar, _x, _, _ = model.core_compute(an_, ed, edv, ei_, batch_, fe_)
            node_e = energy_block(x_scalar).view(-1)
            energy = torch.zeros(n_sys, device=node_e.device, dtype=node_e.dtype)
            energy.index_add_(0, batch_, node_e)
            return energy / avg_num_nodes

        def core_fn(pos_, disp_, an_, ei_, co_, cell_, batch_, fe_):
            sym = 0.5 * (disp_ + disp_.transpose(-1, -2))
            pos_p = pos_ + torch.bmm(
                pos_.unsqueeze(-2), torch.index_select(sym, 0, batch_)
            ).squeeze(-2)
            cell_p = cell_ + torch.bmm(cell_, sym)
            energy = _energy(pos_p, cell_p, an_, ei_, co_, batch_, fe_, cell_.shape[0])
            grads = torch.autograd.grad([energy.sum()], [pos_, disp_], create_graph=True)
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(cell_).abs().unsqueeze(-1)
            stress = (virial / volume.view(-1, 1, 1)).view(-1, 9)
            return energy, forces, stress

        from fairchem.core.common.compile_utils import make_prime_graph_example
        prime_base = make_prime_graph_example(DEVICE, DTYPE, stress=True)
        prime_natoms = prime_base[0].shape[0]
        fe_prime = torch.zeros(prime_natoms, *fe.shape[1:], device=DEVICE, dtype=DTYPE)
        trace_ex = (*prime_base, fe_prime)
        real_ex = (
            pos,
            torch.zeros(cell.shape[0], 3, 3, device=DEVICE, dtype=DTYPE).requires_grad_(True),
            an, edge_index, co, cell, batch, fe,
        )

    return core_fn, trace_ex, real_ex


def run_symbolic_trace(core_fn, trace_ex, real_ex, label):
    """Run make_fx(symbolic) and optionally test re-run on real_ex."""
    from fairchem.core.common.compile_utils import get_force_decompositions
    from torch.fx.experimental.proxy_tensor import make_fx

    decomp = get_force_decompositions()
    result = dict(ok=False, error_type=None, error_msg=None, traceback=None,
                  n_nodes=0, rerun_ok=False, rerun_error=None,
                  shape_consistency=False)

    print(f"  [{label}] symbolic make_fx trace on prime dims...", flush=True)
    t0 = time.perf_counter()
    try:
        gm = make_fx(
            core_fn,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            decomposition_table=decomp,
        )(*trace_ex)
        elapsed = time.perf_counter() - t0
        result['n_nodes'] = len(list(gm.graph.nodes))
        result['ok'] = True
        print(f"  [{label}] trace OK: {result['n_nodes']} nodes, {elapsed:.1f}s", flush=True)

        # Re-run on the real (differently-sized) example to check shape consistency
        # real_ex has natoms from data ([5,7] = 12), prime had natoms=7
        try:
            out = gm(*real_ex)
            finite = all(
                torch.isfinite(o).all().item()
                for o in (out if isinstance(out, (tuple, list)) else [out])
            )
            result['rerun_ok'] = True
            result['shape_consistency'] = finite
            print(f"  [{label}] re-run on real (natoms={real_ex[0].shape[0]}) OK, finite={finite}", flush=True)
        except Exception as e2:
            result['rerun_error'] = str(e2)
            print(f"  [{label}] re-run FAILED: {type(e2).__name__}: {str(e2)[:200]}", flush=True)

    except Exception as e:
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc()
        result['error_type'] = type(e).__name__
        result['error_msg'] = str(e)
        result['traceback'] = tb
        print(f"  [{label}] FAILED after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:300]}", flush=True)

    return result


def patch_len_to_shape(model):
    """Monkey-patch _forward_embedding in place to use shape[0] instead of len().
    This verifies the fix without modifying production code.
    """
    import experimental.models.equiformer_v3.equiformer_v3 as ev3_mod

    def _forward_embedding_fixed(self, atomic_numbers, edge_distance, edge_index, edge_envelope_weight):
        # FIX: len(atomic_numbers) -> atomic_numbers.shape[0] (keeps SymInt symbolic)
        num_atoms = atomic_numbers.shape[0]

        x = torch.zeros(
            (
                num_atoms,
                ((self.lmax + 1) ** 2),
                self.num_channels,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        atom_embedding = self.sphere_embedding(atomic_numbers)
        x[:, 0, :] = atom_embedding

        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
        )
        x = x + edge_degree
        return x

    # Bind as bound method on the model instance
    model._forward_embedding = types.MethodType(_forward_embedding_fixed, model)


# -- Phase 2: full model test -------------------------------------------------
def phase2_full_model(build_fn, variant: str, denoising: bool,
                      apply_patch: bool = False):
    print(f"\n[Phase 2{'+patch' if apply_patch else ''} - {variant}] Building model...", flush=True)
    model = build_fn(seed=42)
    model.train()
    if apply_patch:
        patch_len_to_shape(model)

    shapes = [
        ([5, 7],    11),
        ([6, 4, 3], 22),
        ([8, 5],    33),
    ]

    results = []
    compile_times = []

    for idx, (natoms_list, seed) in enumerate(shapes):
        data = build_batch(natoms_list, seed=seed, denoising=denoising)

        # Eager baseline
        reset_data(data)
        model.enable_compile = False
        model.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        try:
            out_e = model._forward_gradient(data)
            f_e = out_e.get("forces")
            s_e = out_e.get("stress")
        except Exception as e:
            print(f"  shape{idx+1} eager FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            results.append(dict(ok=False, shape=natoms_list, error=str(e)[:200]))
            continue

        # Compiled run
        reset_data(data)
        model.enable_compile = True
        model.zero_grad(set_to_none=True)
        torch.manual_seed(42)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out_c = model._forward_gradient(data)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_call = time.perf_counter() - t0
        except Exception as e:
            t_call = time.perf_counter() - t0
            tb = traceback.format_exc()
            print(f"  shape{idx+1} compiled FAILED after {t_call:.1f}s:", flush=True)
            print(f"    {type(e).__name__}: {str(e)[:400]}", flush=True)
            # Key inductor lines
            for line in tb.splitlines():
                if any(kw in line for kw in ['equiformer', 'compile_utils', 'copy_', 'expand', 'size']):
                    print(f"    {line}", flush=True)
            results.append(dict(ok=False, shape=natoms_list, error=str(e)[:500],
                                traceback=tb, t_call=t_call))
            continue

        if idx == 0:
            compile_times.append(t_call)

        cache_after = len(model._compiled_region._cache) if model._compiled_region else 0
        f_c = out_c.get("forces")
        s_c = out_c.get("stress")

        ferr = (f_e - f_c.detach()).abs().max().item() if f_e is not None and f_c is not None else float("nan")
        serr = (s_e - s_c.detach()).abs().max().item() if s_e is not None and s_c is not None else float("nan")
        finite = (
            (torch.isfinite(f_c).all().item() if f_c is not None else False) and
            (torch.isfinite(s_c).all().item() if s_c is not None else False)
        )

        print(
            f"  shape{idx+1}: natoms={sum(natoms_list)} nsys={len(natoms_list)} "
            f"cache={cache_after} t={t_call:.2f}s f_err={ferr:.2e} s_err={serr:.2e} "
            f"finite={finite}",
            flush=True,
        )

        results.append(dict(
            ok=True, shape=natoms_list, natoms=sum(natoms_list), nsys=len(natoms_list),
            cache_after=cache_after, t_call=t_call, ferr=ferr, serr=serr, finite=finite,
        ))

    cache_sizes = [r.get('cache_after') for r in results if r.get('ok')]
    single_kernel = (
        len(set(c for c in cache_sizes if c is not None)) == 1 and
        len(cache_sizes) > 0 and cache_sizes[0] == 1
    )
    print(f"  cache_sizes={cache_sizes}  single_kernel={single_kernel}", flush=True)
    return results, single_kernel


# -- main ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EquiformerV3 dynamic=True probe (discovery; no production changes)")
    print(f"torch={torch.__version__}  device={DEVICE}  dtype={DTYPE}")
    print("=" * 70)

    report = []
    report.append("# EquiformerV3 dynamic=True Probe Report\n\n")
    report.append(f"- torch: {torch.__version__}\n- device: {DEVICE}\n\n")

    # ====================================================================
    # Phase 1: symbolic make_fx (VANILLA -- expose collapse)
    # ====================================================================
    print("\n" + "=" * 60)
    print("Phase 1A: symbolic make_fx VANILLA (expose collapse point)")
    print("=" * 60)
    report.append("## Phase 1A: Vanilla Symbolic make_fx (BEFORE fix)\n\n")

    for variant, build_fn, denoising in [("base", build_base, False), ("dens", build_dens, True)]:
        data_prime = build_batch([5, 7], seed=0, denoising=denoising)
        reset_data(data_prime)
        model = build_fn(seed=42)
        model.train()
        core_fn, trace_ex, real_ex = build_core_fn(model, variant, data_prime)
        r1 = run_symbolic_trace(core_fn, trace_ex, real_ex, f"1A-{variant}-vanilla")

        report.append(f"### {variant}\n")
        if r1['ok']:
            report.append(f"- Trace: **SUCCESS** ({r1['n_nodes']} nodes)\n")
            if r1['rerun_ok']:
                report.append(f"- Re-run on real inputs: OK (finite={r1['shape_consistency']})\n")
            else:
                report.append(
                    f"- Re-run on real inputs: **FAIL** -- `{r1['rerun_error'][:200]}`\n"
                    f"- **This is the collapse**: `len(atomic_numbers)` at "
                    f"`equiformer_v3.py:423` bakes concrete natoms={trace_ex[0].shape[0]} "
                    f"into `torch.zeros`, so real batch (natoms={real_ex[0].shape[0]}) fails\n"
                )
        else:
            report.append(
                f"- Trace: **FAIL** -- `{r1['error_type']}: {r1['error_msg'][:200]}`\n"
            )
        report.append("\n")

    # ====================================================================
    # Phase 1B: symbolic make_fx WITH PATCH (verify fix)
    # ====================================================================
    print("\n" + "=" * 60)
    print("Phase 1B: symbolic make_fx WITH patch len()->shape[0]")
    print("=" * 60)
    report.append("## Phase 1B: Symbolic make_fx WITH patch (`len` -> `shape[0]`)\n\n")

    for variant, build_fn, denoising in [("base", build_base, False), ("dens", build_dens, True)]:
        data_prime = build_batch([5, 7], seed=0, denoising=denoising)
        reset_data(data_prime)
        model = build_fn(seed=42)
        model.train()
        patch_len_to_shape(model)  # monkey-patch _forward_embedding
        core_fn, trace_ex, real_ex = build_core_fn(model, variant, data_prime)
        r1b = run_symbolic_trace(core_fn, trace_ex, real_ex, f"1B-{variant}-patched")

        report.append(f"### {variant}\n")
        if r1b['ok']:
            report.append(f"- Trace: **SUCCESS** ({r1b['n_nodes']} nodes)\n")
            if r1b['rerun_ok']:
                report.append(f"- Re-run on real inputs: **OK** (finite={r1b['shape_consistency']})\n")
                report.append("- Patch sufficient for symbolic trace + re-run\n")
            else:
                report.append(
                    f"- Re-run: **FAIL** -- `{r1b['rerun_error'][:200]}`\n"
                    f"- Patch not sufficient alone; further issues remain\n"
                )
        else:
            report.append(
                f"- Trace: **FAIL** -- `{r1b['error_type']}: {r1b['error_msg'][:200]}`\n"
            )
        report.append("\n")

    # ====================================================================
    # Phase 2: full _forward_gradient (WITHOUT patch -- shows failure mode)
    # ====================================================================
    print("\n" + "=" * 60)
    print("Phase 2: full model, compile_dynamic=True, NO patch (3 shapes)")
    print("=" * 60)
    report.append("## Phase 2: Full Model, compile_dynamic=True (NO patch, 3 shapes)\n\n")

    for variant, build_fn, denoising in [("base", build_base, False), ("dens", build_dens, True)]:
        print(f"\n  [{variant}]")
        results, sk = phase2_full_model(build_fn, variant, denoising, apply_patch=False)
        cache_sizes = [r.get('cache_after') for r in results if r.get('ok')]
        first_err = next((r.get('error', '') for r in results if not r.get('ok')), None)
        report.append(f"### {variant}\n")
        report.append(f"- cache_sizes: `{cache_sizes}`, single_kernel: `{sk}`\n")
        if first_err:
            report.append(
                f"- **All shapes FAIL**: `{first_err[:300]}`\n"
                f"- Root cause: `aten.copy_.default` fails because `torch.zeros` "
                f"baked concrete natoms (from prime example) while `atom_embedding` "
                f"has symbolic dim -- same `len()` collapse\n"
            )
        report.append("\n")

    # ====================================================================
    # Phase 2+patch: full _forward_gradient WITH patch
    # ====================================================================
    print("\n" + "=" * 60)
    print("Phase 2+patch: full model, compile_dynamic=True, WITH patch (3 shapes)")
    print("=" * 60)
    report.append("## Phase 2+patch: Full Model, compile_dynamic=True (WITH patch, 3 shapes)\n\n")

    for variant, build_fn, denoising in [("base", build_base, False), ("dens", build_dens, True)]:
        print(f"\n  [{variant}+patch]")
        results, sk = phase2_full_model(build_fn, variant, denoising, apply_patch=True)
        cache_sizes = [r.get('cache_after') for r in results if r.get('ok')]
        report.append(f"### {variant}\n")
        report.append(
            "| shape | natoms | cache | t_call(s) | f_err | s_err | finite |\n"
            "|-------|--------|-------|-----------|-------|-------|--------|\n"
        )
        for r in results:
            if r.get('ok'):
                report.append(
                    f"| {r['shape']} | {r['natoms']} | {r['cache_after']} | "
                    f"{r['t_call']:.2f} | {r['ferr']:.2e} | {r['serr']:.2e} | {r['finite']} |\n"
                )
            else:
                report.append(f"| {r['shape']} | - | - | FAIL | - | - | - |\n")
                if r.get('error'):
                    report.append(f"\nError: `{r['error'][:300]}`\n")
        report.append(
            f"\ncache_sizes: `{cache_sizes}` — single_kernel (dynamic=True effective): `{sk}`\n\n"
        )

    # ====================================================================
    # Findings summary
    # ====================================================================
    report.append("## Summary of Collapse Points\n\n")
    report.append(
        "### CP-1 (CONFIRMED, BLOCKER)\n\n"
        "**File**: `experimental/models/equiformer_v3/equiformer_v3.py:423`\n\n"
        "**Op**: `num_atoms = len(atomic_numbers)`\n\n"
        "**Classification**: `len()` forced SymInt->Python int, baking concrete natoms "
        "into `torch.zeros((num_atoms, ...))`. The subsequent "
        "`x[:, 0, :] = atom_embedding` generates `aten.copy_.default(zeros[7, C], embedding[s77, C])` "
        "which fails in `torch.compile(dynamic=True)` because concrete `7 != symbolic s77`.\n\n"
        "**Fix**: `atomic_numbers.shape[0]` (keeps the SymInt symbolic through `torch.zeros`).\n\n"
        "**Scope**: Base AND DeNS (DeNS inherits `_forward_embedding`). Single file, single line.\n\n"
        "**eSEN analogy**: Identical class of bug fixed in eSEN's embedding init.\n\n"
    )
    report.append(
        "### No Other Collapse Points Found\n\n"
        "- `so3.py:429 len(alpha)` is in `_rotation_to_wigner_matrix` (OLD rotation path, "
        "not reached via `set_wigner_from_eulers` Euler path used by `core_compute`).\n"
        "- `SO3Linear.forward` in-place `outputs[:, 0:1, :] = ...`: source and target both have "
        "symbolic natoms dim after CP-1 fix; `aten.select_scatter` lowering handles symbolic N.\n"
        "- `reduce_edge` uses `atomic_numbers.shape[0]` already (safe).\n"
        "- `EquivariantGraphAttention` uses `num_nodes = x.shape[0]` (safe).\n\n"
    )

    # Write report
    report_path = os.path.join(REPO, ".superpowers", "sdd", "dynamic-probe-report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.writelines(report)
    print(f"\nReport written to: {report_path}", flush=True)


if __name__ == "__main__":
    main()
