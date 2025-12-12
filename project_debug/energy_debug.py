#!/usr/bin/env python
"""
energy_debug.py

Sanity check:
  - Build the conformer env from the ai4mat23/mlp_torchani config
  - Sample random internal DOFs (torsions + bond lengths + bond angles)
  - Push them through RDKitConformer
  - Evaluate TorchANI energy
  - Compare to a reference (all-zero internal state)
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from gflownet.envs.conformers.conformer import Conformer


def build_env_from_configs():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    config_dir = repo_root / "config"

    exp_cfg_path = config_dir / "experiments" / "ai4mat23" / "mlp_torchani.yaml"
    base_env_path = config_dir / "env" / "conformers" / "conformer.yaml"

    print(f"Here: {here}")
    print(f"Experiment config: {exp_cfg_path}")
    print(f"Base env config:   {base_env_path}")

    exp_cfg = OmegaConf.load(exp_cfg_path)
    base_env_cfg = OmegaConf.load(base_env_path)

    print("\n=== FULL EXPERIMENT CONFIG (mlp_torchani.yaml) ===")
    print(OmegaConf.to_yaml(exp_cfg))

    env_overrides = exp_cfg.get("env", {})
    env_cfg = OmegaConf.merge(base_env_cfg, env_overrides)

    # Drop Hydra-specific keys
    env_cfg = env_cfg.copy()
    if "defaults" in env_cfg:
        env_cfg.pop("defaults")

    print("\n=== ENV CONFIG PASSED TO Conformer ===")
    print(OmegaConf.to_yaml(env_cfg))

    env = instantiate(env_cfg)  # Conformer
    return env, env_cfg, exp_cfg


def build_torchani_proxy():
    """
    Instantiate the TorchANI proxy used in the ai4mat23 experiment.

    In mlp_torchani.yaml we have:
        - override /proxy: conformers/torchani

    which corresponds to the file:
        config/proxy/conformers/torchani.yaml

    Here we load that file directly and instantiate it.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    config_dir = repo_root / "config"

    proxy_cfg_path = config_dir / "proxy" / "conformers" / "torchani.yaml"
    print(f"\nProxy config: {proxy_cfg_path}")

    proxy_cfg = OmegaConf.load(proxy_cfg_path)

    print("\n=== PROXY CONFIG (torchani.yaml) ===")
    print(OmegaConf.to_yaml(proxy_cfg))

    proxy = instantiate(
        proxy_cfg,
        device="cpu",          # or "cuda" / "mps" if you want and it works
        float_precision="f32",
    )

    return proxy


def energy_from_state(env: Conformer, proxy, state):
    """
    Compute TorchANI energy for a single torus state (1D array of length env.n_dim).
    """
    # Sync geometry
    env.sync_conformer_with_state(state.tolist())

    # Build proxy input: (n_atoms, 4) → (1, n_atoms, 4)
    proxy_state = env.statebatch2proxy([state])
    proxy_state = torch.from_numpy(proxy_state).float()  # shape (1, n_atoms, 4)

    with torch.no_grad():
        energy = proxy(proxy_state)  # shape (1,) or (1, 1) depending on impl

    return float(energy.squeeze().cpu().numpy())


def main():
    # 1) Build env + exp config
    env, env_cfg, exp_cfg = build_env_from_configs()

    print("\n=== Conformer env ===")
    print("Type:", type(env))
    print("SMILES:", env.smiles)
    print("n_dim (internal DOFs):", env.n_dim)
    print(f"  -> torsions:     {env.n_torsion_angles}")
    print(f"  -> bond lengths: {env.n_bond_lengths}")
    print(f"  -> bond angles:  {env.n_bond_angles}")

    # 2) Build TorchANI proxy
    proxy = build_torchani_proxy()

    n_dim = env.n_dim

    # Reference state: all zeros (all internal DOFs at reference)
    ref_state = np.zeros(n_dim, dtype=float)
    ref_energy = energy_from_state(env, proxy, ref_state)
    print(f"\nReference energy (all zeros)   : {ref_energy:.4f}")

    # 3) Small perturbation around reference
    rng = np.random.default_rng(seed=0)
    small_delta = rng.normal(loc=0.0, scale=0.1, size=n_dim)  # gentle wiggle
    small_state = ref_state + small_delta
    small_energy = energy_from_state(env, proxy, small_state)
    print(f"Small-perturbed energy (scale=0.1): {small_energy:.4f}")

    # 4) Large perturbation
    large_delta = rng.normal(loc=0.0, scale=0.5, size=n_dim)  # stronger distortion
    large_state = ref_state + large_delta
    large_energy = energy_from_state(env, proxy, large_state)
    print(f"Large-perturbed energy (scale=0.5): {large_energy:.4f}")

    print("\nEnergy differences:")
    print(f"  ΔE(small - ref) = {small_energy - ref_energy:.4f}")
    print(f"  ΔE(large - ref) = {large_energy - ref_energy:.4f}")

    # ------------------------------------------------------------------
    # Optional: Boltzmann-style reward sanity check
    # ------------------------------------------------------------------

    # Use the same beta as in the env config, fallback to 32 if missing
    beta = float(getattr(env, "reward_beta", env_cfg.get("reward_beta", 32.0)))

    energies = np.array([ref_energy, small_energy, large_energy], dtype=np.float64)
    labels = ["ref", "small", "large"]

    # Shift by minimum energy to avoid huge exponents
    E_min = energies.min()
    shifted = energies - E_min

    rewards = np.exp(-beta * shifted)
    rewards_rel = rewards / rewards.max()

    print("\nBoltzmann-style rewards (unnormalized):")
    for lbl, E, r, rs in zip(labels, energies, rewards, rewards_rel):
        print(f"  {lbl:6s}  E = {E: .6f}   reward = {r:.3e}   rel = {rs:.3f}")



if __name__ == "__main__":
    main()
