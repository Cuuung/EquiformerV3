"""Task-4 hard gate (DeNS): eager vs compiled conservative ``_forward_gradient``
(force + stress double-backward, with the EAGER dens_block denoising head) must
align gradients.

Uses a denoising-active batch (mixed noise mask, stress live) so EVERY trainable
param — backbone, energy_block, the SO3Linear force_embedding (passed into the
compiled region as an explicit ``fe`` input), and the eager dens_block — receives
a gradient. Asserts per-param cos > 0.999, norm-ratio within 5%, compiled grad-set
== eager grad-set AND all params get grad (the strip_detach silent-severing
check), plus a changed-shape re-trace. sys.exit(1) on any failure.

Run:
    bash scripts/compile_env.sh compile_dev/test_stage_conservative_dens.py
"""

from __future__ import annotations

import sys

from _conservative_common import build_dens, run_full_gate


def main():
    ok = run_full_gate(build_dens, label="DeNS", denoising=True)
    if not ok:
        print("\nGATE: FAIL")
        sys.exit(1)
    print("\nGATE: PASS")


if __name__ == "__main__":
    main()
