"""Minimal single-point energy/force test for the EquiformerV3 MPtrj checkpoint.

Usage:
    python single_point_test.py \
        --checkpoint checkpoints/checkpoint/mptrj_gradient.pt \
        --device cuda
"""

import argparse

import numpy as np
from ase.build import bulk

from fairchem.core import OCPCalculator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint/mptrj_gradient.pt",
        help="Path to the EquiformerV3 checkpoint.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    calc = OCPCalculator(
        checkpoint_path=args.checkpoint,
        cpu=(args.device == "cpu"),
        seed=0,
    )
    # Disable AMP scaler for clean fp32 inference (matches eval scripts).
    calc.trainer.scaler = None

    # A small periodic test structure: 2-atom NaCl primitive cell.
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print("=" * 60)
    print("Structure:", atoms.get_chemical_formula(), "| n_atoms =", len(atoms))
    print("Cell volume (A^3):", round(atoms.get_volume(), 4))
    print("-" * 60)
    print(f"Total energy (eV)      : {energy:.6f}")
    print(f"Energy per atom (eV)   : {energy / len(atoms):.6f}")
    print("Forces (eV/A):")
    print(np.array2string(forces, precision=6, suppress_small=True))
    print(f"Max |force| (eV/A)     : {np.abs(forces).max():.6e}")
    try:
        stress = atoms.get_stress()
        print("Stress (Voigt, eV/A^3):")
        print(np.array2string(stress, precision=6, suppress_small=True))
    except Exception as exc:  # model may not output stress
        print(f"Stress: not available ({type(exc).__name__})")
    print("=" * 60)


if __name__ == "__main__":
    main()
