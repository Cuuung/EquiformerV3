"""Task-4 hard gate (base / non-DeNS): eager vs compiled conservative
``_forward_gradient`` (force + stress double-backward) must align gradients.

Judges GRADIENT ALIGNMENT (per-param cos > 0.999, norm-ratio within 5%, every
trainable param receives a gradient, compiled grad-set == eager grad-set) — the
strip_detach silent-severing check — plus numeric equivalence and a changed-shape
re-trace. NOT loss trajectories (CUDA atomics make those noisy). sys.exit(1) on
any failure.

Run:
    bash scripts/compile_env.sh compile_dev/test_stage_conservative_base.py
"""

from __future__ import annotations

import sys

from _conservative_common import build_base, run_full_gate


def main():
    ok = run_full_gate(build_base, label="base", denoising=False)
    if not ok:
        print("\nGATE: FAIL")
        sys.exit(1)
    print("\nGATE: PASS")


if __name__ == "__main__":
    main()
