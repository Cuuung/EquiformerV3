"""Verify core_compute refactor is numerically lossless for both DeNS and base V3.

TDD flow:
  Step 1 (BEFORE refactor): record baseline outputs
    bash scripts/compile_env.sh compile_dev/verify_core_compute_refactor.py --record

  Step 2 (AFTER refactor):  compare to baseline, assert diff < 1e-6
    bash scripts/compile_env.sh compile_dev/verify_core_compute_refactor.py

Gate discipline: every tolerance is enforced with sys.exit(1) on failure.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
from torch_geometric.data import Data

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "_core_baseline.pt")
TOL = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Small model configs (direct_prediction=False => gradient path)
# ─────────────────────────────────────────────────────────────────────────────

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
)


def build_base_model(seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
    torch.manual_seed(seed)
    return EquiformerV3_OC(**_SMALL_CFG).eval()


def build_dens_model(seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC
    torch.manual_seed(seed)
    return EquiformerV3DeNS_OC(**_SMALL_CFG).eval()


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


def run_gradient(model, data: Data) -> dict:
    """Call _forward_gradient on a deep-copy so data.pos side-effects don't carry over."""
    d = copy.deepcopy(data)
    out = model._forward_gradient(d)
    return {k: v.detach().double() for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record", action="store_true",
        help="Record pre-refactor baseline to _core_baseline.pt"
    )
    args = parser.parse_args()

    data = build_batch(n=8, seed=123)

    if args.record:
        print("=" * 60)
        print("MODE: --record  (saving pre-refactor baseline)")
        print("=" * 60)

        base_model = build_base_model(seed=42)
        dens_model = build_dens_model(seed=42)

        with torch.enable_grad():
            base_out = run_gradient(base_model, data)
            dens_out = run_gradient(dens_model, data)

        torch.save({"base": base_out, "dens": dens_out}, BASELINE_PATH)
        print(f"Saved baseline -> {BASELINE_PATH}")
        print("  base outputs:")
        for k, v in base_out.items():
            print(f"    {k}: shape={tuple(v.shape)}  first_vals={v.flatten()[:3].tolist()}")
        print("  dens outputs:")
        for k, v in dens_out.items():
            print(f"    {k}: shape={tuple(v.shape)}  first_vals={v.flatten()[:3].tolist()}")
        return

    # ── Gate: compare refactored outputs to baseline ──────────────────────────
    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: baseline not found at {BASELINE_PATH}", file=sys.stderr)
        print("Run with --record first.", file=sys.stderr)
        sys.exit(1)

    baseline = torch.load(BASELINE_PATH, weights_only=True)

    base_model = build_base_model(seed=42)
    dens_model = build_dens_model(seed=42)

    with torch.enable_grad():
        base_out = run_gradient(base_model, data)
        dens_out = run_gradient(dens_model, data)

    print("=" * 60)
    print("MODE: gate  (verify core_compute refactor losslessness)")
    print("=" * 60)

    FAIL = False

    for label, out, bl_key in [("base", base_out, "base"), ("dens", dens_out, "dens")]:
        bl = baseline[bl_key]
        for k in ("energy", "forces", "stress"):
            if k not in bl:
                continue
            diff = (out[k] - bl[k].double()).abs().max().item()
            ok = diff < TOL
            status = "OK" if ok else "FAIL"
            print(f"  [{label}] {k}: max_diff={diff:.3e}  {status}")
            if not ok:
                print(
                    f"ASSERT FAILED: [{label}] {k} max_diff={diff:.3e} >= TOL={TOL:.0e}",
                    file=sys.stderr,
                )
                FAIL = True

    print("=" * 60)
    if FAIL:
        print("GATE: FAIL")
        sys.exit(1)
    else:
        print("GATE: PASS  — core_compute refactor is numerically lossless (<1e-6)")


if __name__ == "__main__":
    main()
