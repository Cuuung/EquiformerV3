"""Task-4 diagnostic: per-parameter gradient tracking, eager vs compiled
conservative ``_forward_gradient``, for both base and DeNS EquiformerV3.

Prints the full per-param cos / norm-ratio table (sorted worst-first) and the
grad-coverage summary so a near-miss (e.g. a single severed module) is visible.
This is a DIAGNOSTIC — it never gates; run the test_stage_conservative_*.py files
for the hard asserts.

Run:
    bash scripts/compile_env.sh compile_dev/diag_grad_tracking.py
"""

from __future__ import annotations

from _conservative_common import (
    build_base,
    build_dens,
    build_batch,
    grad_align,
    grads_at,
    warmup_compiled,
)


def diag(build_fn, label: str, denoising: bool):
    print("=" * 70)
    print(f"diag grad tracking [{label}] denoising={denoising}")
    print("=" * 70)
    model = build_fn(seed=42)
    data = build_batch([6, 7], seed=123, denoising=denoising)
    n_total = sum(1 for _, p in model.named_parameters() if p.requires_grad)

    warmup_compiled(model, data)
    out_e, ge = grads_at(model, data, compiled=False)
    out_c, gc = grads_at(model, data, compiled=True)

    eerr = (out_e["energy"] - out_c["energy"]).abs().max().item()
    ferr = (out_e["forces"] - out_c["forces"]).abs().max().item()
    serr = (out_e["stress"] - out_c["stress"]).abs().max().item() if "stress" in out_e else 0.0
    print(f"energy err={eerr:.2e}  force err={ferr:.2e}  stress err={serr:.2e}")

    m = grad_align(ge, gc)
    print(f"grad coverage: eager {len(ge)}/{n_total}  compiled {len(gc)}/{n_total}  "
          f"eager-grad covered by compiled={set(ge) <= set(gc)}")
    print(f"concat cos={m['gcos']:.6f} ratio={m['gratio']:.5f}  |  "
          f"per-param(|g|>1e-4*max, n={len(m['big'])}) "
          f"min cos={m['pmin_cos']:.6f} max|1-ratio|={m['pdev_ratio']:.3e}")
    print(f"{'param':<48}{'cos':>10}{'ratio':>10}{'|ge|':>12}")
    for k, c, r, na in sorted(m["rows"], key=lambda x: x[1]):
        print(f"{k:<48}{c:>10.6f}{r:>10.4f}{na:>12.2e}")
    print()


def main():
    diag(build_base, "base", denoising=False)
    diag(build_dens, "DeNS", denoising=True)


if __name__ == "__main__":
    main()
