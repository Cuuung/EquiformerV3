"""Gate: assert _forward_edge is called exactly ONCE per forward (finding-A fix).

Before the fix, _forward_direct and _forward_gradient each called _forward_edge
twice per forward (once inside core_compute, once redundantly for the heads).
After the fix, each forward calls _forward_edge exactly once.

Usage:
    bash scripts/compile_env.sh compile_dev/test_single_frame.py

Gate discipline: sys.exit(1) on failure.
"""

from __future__ import annotations

import sys
import types
import copy

import torch
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# Small model config
# ─────────────────────────────────────────────────────────────────────────────

_SMALL_CFG = dict(
    use_pbc=False,
    otf_graph=True,
    regress_forces=True,
    regress_stress=True,
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
)


def build_base_model(direct: bool, seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
    torch.manual_seed(seed)
    cfg = dict(_SMALL_CFG, direct_prediction=direct)
    return EquiformerV3_OC(**cfg).eval()


def build_dens_model(direct: bool, seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC
    torch.manual_seed(seed)
    cfg = dict(_SMALL_CFG, direct_prediction=direct)
    return EquiformerV3DeNS_OC(**cfg).eval()


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


# ─────────────────────────────────────────────────────────────────────────────
# Counter monkeypatch
# ─────────────────────────────────────────────────────────────────────────────

class _CallCounter:
    """Wraps model._forward_edge and counts calls."""
    def __init__(self, model):
        self.model = model
        self.count = 0
        self._orig = model._forward_edge

    def __enter__(self):
        counter = self

        def _counted(self_m, edge_distance, edge_distance_vec):
            counter.count += 1
            return counter._orig(edge_distance, edge_distance_vec)

        self.model._forward_edge = types.MethodType(_counted, self.model)
        return self

    def __exit__(self, *args):
        self.model._forward_edge = self._orig


def count_forward_edge_calls(model, forward_fn, data):
    """Run forward_fn(data) with a call counter on _forward_edge; return count."""
    with _CallCounter(model) as ctr:
        d = copy.deepcopy(data)
        with torch.enable_grad():
            forward_fn(d)
        return ctr.count


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data = build_batch(n=8, seed=123)
    FAIL = False

    print("=" * 60)
    print("Gate: single rotation frame per forward (finding-A fix)")
    print("=" * 60)

    checks = [
        # (label, model_builder_args, forward_method)
        ("base _forward_direct  (direct=True)",
         lambda: build_base_model(direct=True),
         "_forward_direct"),
        ("base _forward_gradient (direct=False)",
         lambda: build_base_model(direct=False),
         "_forward_gradient"),
        ("dens _forward_direct  (direct=True)",
         lambda: build_dens_model(direct=True),
         "_forward_direct"),
        ("dens _forward_gradient (direct=False, regress_stress=False)",
         lambda: build_dens_model(direct=False),
         "_forward_gradient"),
    ]

    for label, model_fn, method in checks:
        model = model_fn()
        fwd = getattr(model, method)
        count = count_forward_edge_calls(model, fwd, data)
        ok = count == 1
        status = "OK" if ok else "FAIL (was {})".format(count)
        print(f"  [{label}]: _forward_edge calls = {count}  {status}")
        if not ok:
            FAIL = True

    # Also verify that forces are finite (model is still functional)
    print()
    print("Sanity: forces are finite")
    for label, model_fn, method in checks:
        model = model_fn()
        fwd = getattr(model, method)
        d = copy.deepcopy(data)
        with torch.enable_grad():
            out = fwd(d)
        if 'forces' in out:
            fin = out['forces'].isfinite().all().item()
            status = "OK" if fin else "FAIL (non-finite forces!)"
            print(f"  [{label}]: forces finite = {fin}  {status}")
            if not fin:
                FAIL = True

    print("=" * 60)
    if FAIL:
        print("GATE: FAIL")
        sys.exit(1)
    else:
        print("GATE: PASS  — _forward_edge called exactly once per forward")


if __name__ == "__main__":
    main()
