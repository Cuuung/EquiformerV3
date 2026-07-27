"""GATE: with the DPA4 switches at their defaults, the model must be BIT-IDENTICAL to before.

Why this exists (handoff doc §5.2): the whole point of the DPA4 ablation is to compare against
an EXISTING baseline. If adding the switches perturbs the default path by even one ULP -- a
different RNG draw order, a reordered construction, an extra module in the tree -- that baseline
silently stops being a baseline and every ablation row becomes uninterpretable.

Two-step TDD flow (same discipline as compile_dev/verify_core_compute_refactor.py):

  Step 1, on the PRE-change commit:
      git stash                                              # or: git checkout <pre-change sha>
      PYTHONPATH=src ./.venv/bin/python tests/experimental/equiformer_v3/verify_dpa4_default_identical.py --record
  Step 2, on the changed tree:
      git stash pop
      PYTHONPATH=src ./.venv/bin/python tests/experimental/equiformer_v3/verify_dpa4_default_identical.py

Step 2 exits non-zero unless EVERY tensor matches with torch.equal (exact, not a tolerance) and
the state_dict key sets are identical.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
from torch_geometric.data import Data

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "_dpa4_default_baseline.pt")

# Attention shape mirrors the N2L2C64 recipe (H=8, Ca=64, Cv=16) so the focus-divisibility and
# per-head bias shapes are the ones the ablation will actually run with.
_SMALL_CFG = dict(
    use_pbc=False,
    otf_graph=True,
    regress_forces=True,
    regress_stress=True,
    direct_prediction=False,
    lmax=2,
    mmax=2,
    num_layers=2,
    num_channels=64,
    num_radial_basis=32,
    max_radius=5.0,
    max_neighbors=20,
    attn_hidden_channels=32,
    num_heads=8,
    attn_alpha_channels=64,
    attn_value_channels=16,
    ffn_hidden_channels=64,
    edge_channels=32,
    attn_grid_resolution_list=[14, 8],
    ffn_grid_resolution_list=[14, 14],
    attn_activation="sep-merge_gates2_swiglu",
    ffn_activation="sep-merge_gates2_swiglu",
    norm_type="merge_layer_norm",
    use_envelope=True,
)


def build_base_model(seed: int = 42):
    from fairchem.experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
    torch.manual_seed(seed)
    return EquiformerV3_OC(**_SMALL_CFG).eval()


def build_dens_model(seed: int = 42):
    from fairchem.experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC
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
    d = copy.deepcopy(data)
    out = model._forward_gradient(d)
    return {k: v.detach().double() for k, v in out.items()}


def snapshot() -> dict:
    """Forward outputs + every parameter, for both model classes, under fixed seeds."""
    data = build_batch(n=8, seed=123)
    snap = {}
    for tag, builder in (("base", build_base_model), ("dens", build_dens_model)):
        model = builder(seed=42)
        with torch.enable_grad():
            snap[f"{tag}/out"] = run_gradient(model, data)
        # Parameter VALUES too: a construction reorder changes RNG consumption without ever
        # showing up in a forward diff if the perturbed module happens to be unused.
        snap[f"{tag}/params"] = {k: v.detach().double().clone() for k, v in model.named_parameters()}
        snap[f"{tag}/no_wd"] = sorted(model.no_weight_decay())
    return snap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true",
                        help="record the PRE-change baseline (run on the pre-change tree)")
    args = parser.parse_args()

    snap = snapshot()

    if args.record:
        torch.save(snap, BASELINE_PATH)
        print(f"recorded baseline -> {BASELINE_PATH}")
        for tag in ("base", "dens"):
            print(f"  {tag}: {len(snap[f'{tag}/params'])} params, "
                  f"outputs={sorted(snap[f'{tag}/out'])}, no_wd={len(snap[f'{tag}/no_wd'])}")
        return

    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: baseline not found at {BASELINE_PATH}; run with --record on the "
              f"pre-change tree first.", file=sys.stderr)
        sys.exit(1)

    base = torch.load(BASELINE_PATH, weights_only=False)
    failures = []

    for tag in ("base", "dens"):
        # --- 1. exact forward equality -------------------------------------------------
        b_out, n_out = base[f"{tag}/out"], snap[f"{tag}/out"]
        if set(b_out) != set(n_out):
            failures.append(f"{tag}: output keys changed {sorted(b_out)} -> {sorted(n_out)}")
        for k in sorted(set(b_out) & set(n_out)):
            if not torch.equal(b_out[k], n_out[k]):
                d = (b_out[k] - n_out[k]).abs().max().item()
                failures.append(f"{tag}/{k}: NOT bit-identical, max|diff|={d:.3e}")
            else:
                print(f"  OK  {tag}/{k:16s} bit-identical  shape={tuple(n_out[k].shape)}")

        # --- 2. state_dict key set unchanged (no new params at defaults) ---------------
        b_p, n_p = base[f"{tag}/params"], snap[f"{tag}/params"]
        added, removed = set(n_p) - set(b_p), set(b_p) - set(n_p)
        if added or removed:
            failures.append(f"{tag}: param keys changed  added={sorted(added)} removed={sorted(removed)}")
        else:
            print(f"  OK  {tag}/params            key set unchanged ({len(n_p)} params)")

        # --- 3. parameter values identical (catches RNG-order drift) ------------------
        bad = [k for k in sorted(set(b_p) & set(n_p)) if not torch.equal(b_p[k], n_p[k])]
        if bad:
            failures.append(f"{tag}: {len(bad)} params changed value, e.g. {bad[:3]}")
        else:
            print(f"  OK  {tag}/params            values bit-identical")

        # --- 4. optimizer routing unchanged -------------------------------------------
        if base[f"{tag}/no_wd"] != snap[f"{tag}/no_wd"]:
            failures.append(f"{tag}: no_weight_decay() set changed -> Muon/AdamW routing differs")
        else:
            print(f"  OK  {tag}/no_weight_decay    unchanged ({len(snap[f'{tag}/no_wd'])} names)")

    if failures:
        print("\nGATE FAIL:", file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        sys.exit(1)
    print("\nGATE PASS: DPA4 switches at their defaults are bit-identical to the pre-change model.")


if __name__ == "__main__":
    main()
