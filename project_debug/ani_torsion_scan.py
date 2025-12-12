#!/usr/bin/env python
import os
import math
import numpy as np
import torch
import torchani

from omegaconf import OmegaConf
from hydra.utils import instantiate

from gflownet.envs.conformers.conformer import Conformer


# -----------------------------
# Helper: load env config & build Conformer
# -----------------------------
def build_env(smiles: str):
    here = os.path.dirname(os.path.abspath(__file__))
    print("Here:", here)

    exp_cfg_path = os.path.join(
        here, "config", "experiments", "ai4mat23", "mlp_torchani.yaml"
    )
    base_env_cfg_path = os.path.join(
        here, "config", "env", "conformers", "conformer.yaml"
    )

    print("Experiment config:", exp_cfg_path)
    print("Base env config:  ", base_env_cfg_path)

    exp_cfg = OmegaConf.load(exp_cfg_path)
    base_env_cfg = OmegaConf.load(base_env_cfg_path)

    # Merge env section from experiment into base env config
    env_cfg = OmegaConf.merge(base_env_cfg, exp_cfg.env)
    env_cfg.smiles = smiles

    print("\n=== FINAL ENV CONFIG (env_cfg) ===")
    print(OmegaConf.to_yaml(env_cfg))

    # Instantiate Conformer via Hydra-style _target_
    env = instantiate(env_cfg)

    print("\n=== Conformer env ===")
    print("Type:", type(env))
    print("SMILES:", env.smiles)
    print("n_dim (torsion DOFs):", env.n_dim)

    return env, env_cfg


# -----------------------------
# Helper: RDKitConformer -> TorchANI inputs
# -----------------------------
def conformer_to_ani_inputs(conf, device):
    """
    Convert RDKitConformer to TorchANI (species, coordinates).

    RDKitConformer here is gflownet.utils.molecule.rdkit_conformer.RDKitConformer.
    It exposes:
      - get_atomic_numbers() -> List[int]
      - get_atom_positions() -> np.ndarray [n_atoms, 3]
    """
    # Atomic numbers, e.g. [6,6,...,1,1]
    Z = conf.get_atomic_numbers()
    R = conf.get_atom_positions()

    Z = torch.tensor(Z, dtype=torch.long, device=device).unsqueeze(0)      # [1, n_atoms]
    R = torch.tensor(R, dtype=torch.float32, device=device).unsqueeze(0)  # [1, n_atoms, 3]

    # Map atomic numbers -> ANI2x species indices
    # ANI2x supports H, C, N, O as {H:0, C:1, N:2, O:3}
    mapping = {1: 0, 6: 1, 7: 2, 8: 3}

    species = torch.empty_like(Z)
    for Zval in Z.unique().tolist():
        if Zval not in mapping:
            raise ValueError(f"Atomic number {Zval} not supported by ANI2x (only H,C,N,O).")
        species[Z == Zval] = mapping[Zval]

    return species, R


# -----------------------------
# Torsion scan
# -----------------------------
def torsion_scan(env: Conformer, ani_model, torsion_idx: int, n_points: int = 73):
    """
    Sweep a single torsion angle (torsion_idx) from -pi to pi and compute
    TorchANI energy + normalized score + Boltzmann reward.
    """
    device = next(ani_model.parameters()).device
    beta = float(getattr(env, "reward_beta", 32.0))  # fall back to 32 if not found

    print(f"\nUsing device: {device}")
    print(f"Using beta (reward_beta): {beta}")

    n_dim = env.n_dim
    base_state = np.zeros(n_dim, dtype=np.float32)

    angles = np.linspace(-math.pi, math.pi, n_points)  # radians
    energies = []

    for a in angles:
        # Copy state, modify only the chosen torsion
        state = base_state.copy()
        state[torsion_idx] = a

        # Update conformer from this state
        env.set_conformer(state.tolist())
        conf = env.conformer

        # Get ANI inputs and energy
        species, R = conformer_to_ani_inputs(conf, device)
        out = ani_model((species, R))
        # out.energies is in Hartree
        E = out.energies.item()
        energies.append(E)

    energies = np.array(energies)

    # Normalize energies to a "score" in [-1, 0]:
    #  - best (minimum) energy -> -1
    #  - worst (maximum) energy -> 0
    E_min = energies.min()
    E_max = energies.max()
    print("\nEnergy range (Hartree):")
    print(f"  E_min = {E_min:.6f}")
    print(f"  E_max = {E_max:.6f}")

    if E_max > E_min:
        # First map to [0,1], then to [-1,0]
        scores = -(energies - E_min) / (E_max - E_min)
    else:
        scores = np.zeros_like(energies)

    # Boltzmann reward with this local score
    rewards = np.exp(-beta * scores)

    return angles, energies, scores, rewards


# -----------------------------
# Main
# -----------------------------
def main():
    # Same SMILES you used in your runs (ibuprofen-like)
    SMILES = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'

    # 1) Build env
    env, env_cfg = build_env(SMILES)

    # 2) Load TorchANI model
    device = torch.device("cpu")
    ani_model = torchani.models.ANI2x().to(device)
    ani_model.eval()

    # 3) Scan the first torsion angle (index 0)
    torsion_idx = 0
    print(f"\nScanning torsion index: {torsion_idx}")
    print(f"Total torsion DOFs (env.n_dim): {env.n_dim}")

    angles, energies, scores, rewards = torsion_scan(
        env, ani_model, torsion_idx=torsion_idx, n_points=73  # ~5° steps
    )

    # 4) Print a small table (sampled points)
    angles_deg = angles * 180.0 / math.pi

    print("\n=== Sampled values (every ~6th point) ===")
    print(f"{'angle_deg':>10}  {'E(Hartree)':>12}  {'score':>10}  {'reward':>12}")
    step = max(1, len(angles_deg) // 12)
    for i in range(0, len(angles_deg), step):
        print(
            f"{angles_deg[i]:10.1f}  "
            f"{energies[i]:12.6f}  "
            f"{scores[i]:10.4f}  "
            f"{rewards[i]:12.3e}"
        )

    # 5) Try plotting if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(angles_deg, energies, marker=".")
        plt.xlabel("Torsion angle (deg)")
        plt.ylabel("Energy (Hartree)")
        plt.title(f"Torsion {torsion_idx} – TorchANI energy")
        plt.grid(True)

        plt.figure()
        plt.plot(angles_deg, rewards, marker=".")
        plt.xlabel("Torsion angle (deg)")
        plt.ylabel("Boltzmann reward")
        plt.title(f"Torsion {torsion_idx} – Reward R = exp(-β·score)")
        plt.yscale("log")  # rewards vary a lot, log scale is easier to read
        plt.grid(True)

        plt.show()
    except ImportError:
        print("\nmatplotlib not installed – skipping plots.")


if __name__ == "__main__":
    main()
