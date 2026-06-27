"""Stage 3 gate: direct-force plain_compile wiring on EquiformerV3 (base + DeNS).

Verifies that wrapping core_compute with plain_compile (enable_compile=True) is
numerically equivalent to eager (enable_compile=False) on the same weights and
the same input.  Also confirms via _compiled_core is not None that compile actually
ran (not a silent eager fallback).

Run:
    bash scripts/compile_env.sh compile_dev/test_stage_direct.py
    ESEN_COMPILE_PROBE=1 bash scripts/compile_env.sh compile_dev/test_stage_direct.py
"""

from __future__ import annotations

import os
import sys

import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Small model config — direct_prediction=True so _forward_direct is called
# ---------------------------------------------------------------------------
_SMALL_CFG = dict(
    use_pbc=False,
    otf_graph=True,
    regress_forces=True,
    regress_stress=False,
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
)

E_TOL = 1e-5
F_TOL = 1e-5


def build_base(enable_compile: bool, seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
    torch.manual_seed(seed)
    return EquiformerV3_OC(**_SMALL_CFG, enable_compile=enable_compile).eval()


def build_dens(enable_compile: bool, seed: int = 42):
    from experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC
    torch.manual_seed(seed)
    return EquiformerV3DeNS_OC(**_SMALL_CFG, enable_compile=enable_compile).eval()


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


def run_forward(model, data: Data) -> dict[str, torch.Tensor]:
    """Run _forward_direct and return detached double tensors."""
    out = model._forward_direct(data)
    return {k: v.detach().double().cpu() for k, v in out.items()}


def check_pair(label: str, eager_out: dict, compiled_out: dict) -> bool:
    """Compare eager vs compiled outputs.  Returns True if PASS."""
    ok = True
    for key in ("energy", "forces", "stress"):
        if key not in eager_out:
            continue
        tol = E_TOL if key == "energy" else F_TOL
        err = (eager_out[key] - compiled_out[key]).abs().max().item()
        status = "OK" if err < tol else "FAIL"
        print(f"  [{label}] {key}: max_diff={err:.3e}  tol={tol:.0e}  {status}")
        if err >= tol:
            print(
                f"ASSERT FAILED: [{label}] {key} max_diff={err:.3e} >= tol={tol:.0e}",
                file=sys.stderr,
            )
            ok = False
    return ok


def main():
    # TF32: high precision matmul (matches recommended compile config)
    torch.set_float32_matmul_precision("high")

    batch = build_batch(n=8, seed=123)

    print("=" * 65)
    print("Stage 3 gate: direct-force plain_compile (base + DeNS)")
    print("=" * 65)

    fail = False

    for variant, build_fn in [("base", build_base), ("DeNS", build_dens)]:
        print(f"\n--- {variant} ---")

        # Build both models with identical weights
        eager_model = build_fn(enable_compile=False, seed=42)
        compiled_model = build_fn(enable_compile=True, seed=42)

        # Warm up the compiled model once (triggers compilation, consumes RNG)
        with torch.no_grad():
            _ = run_forward(compiled_model, batch)
        torch._dynamo.reset()

        # Now run with controlled seed so random gamma is identical
        with torch.no_grad():
            torch.manual_seed(99)
            eager_out = run_forward(eager_model, batch)

            torch.manual_seed(99)
            compiled_out = run_forward(compiled_model, batch)

        compiled_core_set = compiled_model._compiled_core is not None
        print(f"  compiled_core wired: {compiled_core_set}")

        if not compiled_core_set:
            print(
                f"ASSERT FAILED: [{variant}] _compiled_core is None — enable_compile not wired",
                file=sys.stderr,
            )
            fail = True

        if not check_pair(variant, eager_out, compiled_out):
            fail = True

    print("\n" + "=" * 65)
    if fail:
        print("GATE: FAIL")
        sys.exit(1)
    else:
        print("GATE: PASS — direct-force compile is numerically equivalent")


if __name__ == "__main__":
    main()
