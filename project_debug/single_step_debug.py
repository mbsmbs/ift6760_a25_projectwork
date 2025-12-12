#!/usr/bin/env python
"""
single_step_debug.py

Small standalone script to:
  1. Load the conformer environment config used by ai4mat23/mlp_torchani
  2. Instantiate gflownet.envs.conformers.conformer.Conformer
  3. Take one random step in internal DOFs (torsions, bond lengths, bond angles)
  4. Show atom coordinates before/after, bond lengths/angles before/after, and RMSD

Run from the project root:
    (confgfn) $ python project_debug/single_step_debug.py
"""

from pathlib import Path

from rdkit.Chem import rdMolTransforms
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate


def build_env_from_configs():
    """Compose the env config (base + experiment overrides) and instantiate Conformer."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    config_dir = repo_root / "config"

    # Paths to the configs we use
    exp_cfg_path = config_dir / "experiments" / "ai4mat23" / "mlp_torchani.yaml"
    base_env_path = config_dir / "env" / "conformers" / "conformer.yaml"

    print(f"Here: {here}")
    print(f"Experiment config: {exp_cfg_path}")
    print(f"Base env config:   {base_env_path}")

    # Load configs
    exp_cfg = OmegaConf.load(exp_cfg_path)
    base_env_cfg = OmegaConf.load(base_env_path)

    # The experiment overrides the base env via the `env:` block
    env_overrides = exp_cfg.get("env", {})
    env_cfg = OmegaConf.merge(base_env_cfg, env_overrides)

    # Drop Hydra-specific keys that are not part of the Conformer signature
    env_cfg = env_cfg.copy()
    if "defaults" in env_cfg:
        env_cfg.pop("defaults")

    print("\n=== ENV CONFIG PASSED TO Conformer ===")
    print(OmegaConf.to_yaml(env_cfg))

    # Hydra-style instantiation using _target_
    env = instantiate(env_cfg)

    return env, env_cfg


def main():
    # -------------------------------------------------------------------------
    # 1. Build env
    # -------------------------------------------------------------------------
    env, env_cfg = build_env_from_configs()

    print("\n=== Conformer env ===")
    print("Type:", type(env))
    print("SMILES:", getattr(env, "smiles", env_cfg.get("smiles", "<unknown>")))
    print("n_dim (internal DOFs):", getattr(env, "n_dim", "<no n_dim>"))

    # Internal DOF split: torsions / bond lengths / bond angles
    n_dim = env.n_dim
    n_tors = getattr(env, "n_torsion_angles", n_dim)
    n_len = getattr(env, "n_bond_lengths", 0)
    n_ang = getattr(env, "n_bond_angles", 0)
    k0 = n_tors
    k1 = k0 + n_len
    k2 = k1 + n_ang

    print(f"  -> torsions:     {n_tors}")
    print(f"  -> bond lengths: {n_len}")
    print(f"  -> bond angles:  {n_ang}")

    # -------------------------------------------------------------------------
    # 2. Initial internal "state" (all DOFs = 0)
    # -------------------------------------------------------------------------
    state_before = np.zeros(n_dim, dtype=float)
    print("\nInitial internal state (all DOFs):", state_before)

    # Atom coordinates before any change
    conf = env.conformer
    pos_before = conf.get_atom_positions()
    print("First 5 atom coords BEFORE:")
    print(pos_before[:5])

    # --- Bond lengths / angles BEFORE (for a few flex DOFs) ---
    if hasattr(env, "flex_bond_lengths"):
        print("\nFlex bond lengths (indices):", env.flex_bond_lengths)
    if hasattr(env, "flex_bond_angles"):
        print("Flex bond angles (triples):", env.flex_bond_angles)

    print("\nSelected bond lengths BEFORE:")
    for (i, j) in getattr(env, "flex_bond_lengths", [])[:3]:
        d = conf.get_bond_length(i, j)
        print(f"  ({i}-{j}) = {d:.3f} Å")

    print("\nSelected bond angles BEFORE:")
    for (i, j, k) in getattr(env, "flex_bond_angles", [])[:3]:
        a = conf.get_angle(i, j, k)
        print(f"  ({i}-{j}-{k}) = {np.degrees(a):.2f}°")

    # -------------------------------------------------------------------------
    # 3. Sample a random step in all internal DOFs and apply to the conformer
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed=0)
    delta = rng.normal(loc=0.0, scale=0.5, size=n_dim)  # small-ish random step

    # Just for debugging, split the delta into parts
    delta_tors = delta[:k0]
    delta_len = delta[k0:k1]
    delta_ang = delta[k1:k2]

    print("\nDelta (all DOFs):", delta)
    print("  -> torsion part:", delta_tors)
    print("  -> length part: ", delta_len)
    print("  -> angle part:  ", delta_ang)

    state_after = state_before + delta
    print("New internal state (all DOFs):", state_after)

    # Sync the RDKit conformer with the new internal state
    env.sync_conformer_with_state(state_after.tolist())

    # -------------------------------------------------------------------------
    # 4. Atom coordinates after update + bond lengths/angles + RMSD
    # -------------------------------------------------------------------------
    conf = env.conformer
    pos_after = conf.get_atom_positions()
    print("\nFirst 5 atom coords AFTER:")
    print(pos_after[:5])

    print("\nSelected bond lengths AFTER:")
    for (i, j) in getattr(env, "flex_bond_lengths", [])[:3]:
        d = conf.get_bond_length(i, j)
        print(f"  ({i}-{j}) = {d:.3f} Å")

    print("\nSelected bond angles AFTER:")
    for (i, j, k) in getattr(env, "flex_bond_angles", [])[:3]:
        a = conf.get_angle(i, j, k)
        print(f"  ({i}-{j}-{k}) = {np.degrees(a):.2f}°")

    # RMSD = sqrt( (1/N) * sum_i ||x_i - y_i||^2 )
    diff = pos_after - pos_before
    per_atom_sq = (diff * diff).sum(axis=1)
    rmsd = float(np.sqrt(per_atom_sq.mean()))
    print(f"\nRMSD between conformers (Å): {rmsd}")


if __name__ == "__main__":
    main()
