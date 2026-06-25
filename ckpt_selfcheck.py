"""Self-check: can the eval's OCPCalculator load OUR trained grad-ft ckpt and
run a relaxation step (conservative forces, fp32)? Mirrors relaxation/run.py."""

import argparse
import numpy as np
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

from fairchem.core import OCPCalculator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    print(f"[selfcheck] loading: {args.checkpoint}")
    calc = OCPCalculator(checkpoint_path=args.checkpoint, cpu=(args.device == "cpu"), seed=0)
    calc.trainer.scaler = None  # fp32, AMP off (REQUIRED for conservative forces)
    print("[selfcheck] OCPCalculator built OK")

    # slightly perturbed rocksalt so relaxation has something to do
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.80)
    atoms.calc = calc

    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    print(f"[selfcheck] single-point OK | E={e0:.6f} eV ({e0/len(atoms):.4f} eV/atom) "
          f"| max|F|={np.abs(f0).max():.4e} eV/A")
    try:
        print(f"[selfcheck] stress (Voigt): {np.array2string(atoms.get_stress(), precision=4)}")
    except Exception as exc:
        print(f"[selfcheck] stress not available ({type(exc).__name__})")

    # mirror the eval: FrechetCellFilter + FIRE, a few steps only
    print("[selfcheck] running 10 FIRE relaxation steps (FrechetCellFilter)...")
    opt = FIRE(FrechetCellFilter(atoms), logfile="-")
    opt.run(fmax=0.02, steps=10)
    e1 = atoms.get_potential_energy()
    print(f"[selfcheck] relax ran OK | E0={e0:.4f} -> E1={e1:.4f} eV (dE={e1-e0:+.4f})")
    print("[selfcheck] ===== PASS: this ckpt loads & relaxes via OCPCalculator =====")


if __name__ == "__main__":
    main()
