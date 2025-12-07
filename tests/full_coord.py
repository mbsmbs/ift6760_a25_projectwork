# -------------------------------
# Quick sanity check for extended intrinsic coordinates
# -------------------------------
import os
import sys
import numpy as np

# Make repo importable when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gflownet.envs.conformers.conformer import Conformer


def main():
    # Create environment with torsions + flexible bonds + flexible angles
    env = Conformer(
        smiles="CC(C(=O)NC)NC(=O)C",   # alanine dipeptide
        n_torsion_angles=2,
        flex_bond_lengths=[[0, 1], [1, 2]],      # example bonds
        flex_bond_angles=[[0, 1, 2], [1, 2, 3]], # example angles
    )

    print("Number of torsions:", env.n_torsion_angles)
    print("Number of flex bond lengths:", env.n_bond_lengths)
    print("Number of flex bond angles:", env.n_bond_angles)
    print("Total intrinsic DOFs (env.n_dim):", env.n_dim)

    # Get initial state as NumPy array
    state = np.array(env.state, dtype=float)  # length = env.n_dim + 1 (includes time)
    print("Initial state vector:", state)

    # Sync RDKit geometry to this state
    conf0 = env.sync_conformer_with_state(state)

    print("\nReference bond lengths:")
    print(env.ref_bond_lengths)

    print("\nReference bond angles:")
    print(env.ref_bond_angles)

    # ----------------------------------------
    # Perturb internal coordinates (not time)
    # ----------------------------------------
    s = state.copy()

    # internal part = [0 : env.n_dim]
    internal_dim = env.n_dim

    # indices for each block
    k0 = env.n_torsion_angles
    k1 = k0 + env.n_bond_lengths
    k2 = k1 + env.n_bond_angles
    assert k2 == internal_dim

    # torsions
    s[0:k0] += 0.10
    # bond lengths
    s[k0:k1] += 0.20
    # bond angles
    s[k1:k2] += 0.30
    # we leave s[internal_dim] (the time coordinate) unchanged

    # Apply update
    conf1 = env.sync_conformer_with_state(s)

    # Read new geometry
    new_lengths = env.conformer.get_bond_length_vector(env.flex_bond_lengths)
    new_angles = env.conformer.get_angle_vector(env.flex_bond_angles)

    print("\nNew bond lengths (after perturbation):")
    print(new_lengths)

    print("\nNew bond angles (after perturbation):")
    print(new_angles)

    print("\nSanity-check complete.")


if __name__ == "__main__":
    main()
