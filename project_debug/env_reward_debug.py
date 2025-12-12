import os
import math
import random
from typing import Tuple, List

import numpy as np
import torch
import torchani
from omegaconf import OmegaConf
from hydra.utils import instantiate


# ---------- Helpers to load config & build env ----------

def load_env_cfg() -> Tuple[object, object]:
    """
    Load experiment + base env config, merge them, and instantiate Conformer.
    Returns (env, env_cfg).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    print("Here:", here)

    exp_cfg_path = os.path.join(here, "config", "experiments", "ai4mat23", "mlp_torchani.yaml")
    base_env_cfg_path = os.path.join(here, "config", "env", "conformers", "conformer.yaml")

    if not os.path.exists(exp_cfg_path):
        raise FileNotFoundError(f"Experiment config not found at {exp_cfg_path}")
    if not os.path.exists(base_env_cfg_path):
        raise FileNotFoundError(f"Base env config not found at {base_env_cfg_path}")

    print("Experiment config:", exp_cfg_path)
    print("Base env config:  ", base_env_cfg_path)

    exp_cfg = OmegaConf.load(exp_cfg_path)
    base_env_cfg = OmegaConf.load(base_env_cfg_path)

    # Merge base env config with experiment's env overrides
    env_cfg = OmegaConf.merge(base_env_cfg, exp_cfg.env)

    print("\n=== FINAL ENV CONFIG ===")
    print(OmegaConf.to_yaml(env_cfg))

    # Instantiate Conformer environment
    env = instantiate(env_cfg)

    print("\n=== Conformer env ===")
    print("Type:", type(env))
    print("SMILES:", env.smiles)
    print("n_dim (torsion DOFs):", env.n_dim)

    return env, env_cfg


# ---------- TorchANI interface helpers ----------

def build_ani_model(device: torch.device):
    """
    Build ANI2x model on the given device.
    """
    print("\nUsing device:", device)
    # This prints the internal resources path (just as a sanity check)
    import torchani.resources  # noqa: F401
    print(torchani.resources.__file__)

    model = torchani.models.ANI2x().to(device)
    model.eval()
    return model


def conformer_to_ani_inputs(conf, model, device: torch.device):
    """
    Convert RDKitConformer -> (species_tensor, coordinates_tensor) for TorchANI.

    - species_tensor: (1, N_atoms) of indices (0..3) corresponding to H, C, N, O
    - coordinates_tensor: (1, N_atoms, 3) in Å
    """
    # Atomic numbers from the RDKitConformer
    atomic_numbers: List[int] = conf.get_atomic_numbers()  # e.g. [6, 6, ..., 1, 1]
    positions = conf.get_atom_positions()                  # shape (N_atoms, 3)

    # Map atomic numbers to element symbols expected by ANI2x
    num_to_symbol = {1: "H", 6: "C", 7: "N", 8: "O"}
    try:
        symbols = "".join(num_to_symbol[z] for z in atomic_numbers)
    except KeyError as e:
        raise ValueError(f"Encountered atomic number not supported by ANI2x: {e}")

    # TorchANI helper: convert species string -> tensor of indices 0..3
    species = model.species_to_tensor(symbols).to(device)  # (N_atoms,)
    species = species.unsqueeze(0)                         # (1, N_atoms)

    R = torch.tensor(positions, dtype=torch.float32, device=device)
    R = R.unsqueeze(0)  # (1, N_atoms, 3)

    return species, R, atomic_numbers


def ani_energy(model, species, R) -> float:
    """
    Compute ANI energy (Hartree) for a single geometry.
    """
    with torch.no_grad():
        out = model((species, R))
        # ANI2x returns an object with .energies of shape (batch,)
        E = out.energies[0].item()
    return E


# ---------- Main reward-debug logic ----------

def sample_random_state(n_dim: int) -> List[float]:
    """
    Sample a random torsion state in [-pi, pi]^n_dim.
    """
    return [(random.random() * 2 * math.pi - math.pi) for _ in range(n_dim)]


def main():
    # Seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # 1) Build env + get beta
    env, env_cfg = load_env_cfg()
    beta = float(env_cfg.reward_beta)
    print("\nUsing reward_beta (beta):", beta)

    # 2) Build TorchANI model
    device = torch.device("cpu")
    model = build_ani_model(device)

    # 3) Sample random torsion states
    n_samples = 100  # you can increase to 500 or 1000 if you want smoother stats
    energies = []
    states = []

    print(f"\nSampling {n_samples} random torsion states in [-pi, pi]^{env.n_dim} ...")

    for i in range(n_samples):
        state = sample_random_state(env.n_dim)
        states.append(state)

        # Update conformer to this state
        # Conformer.set_conformer(state) will internally call sync_conformer_with_state(state)
        env.set_conformer(state)
        conf = env.conformer

        species, R, _ = conformer_to_ani_inputs(conf, model, device)
        E = ani_energy(model, species, R)
        energies.append(E)

    energies = np.array(energies)
    E_min = float(energies.min())
    E_max = float(energies.max())
    dE = max(E_max - E_min, 1e-8)

    print("\nEnergy stats over sampled states (Hartree):")
    print(f"  E_min = {E_min:.9f}")
    print(f"  E_max = {E_max:.9f}")
    print(f"  ΔE    = {dE:.9f}")

    # 4) Map energies -> normalized scores in [-1, 0], then -> rewards
    scores = -(energies - E_min) / dE        # minima -> ~-1, maxima -> ~0
    rewards = np.exp(-beta * scores)        # Boltzmann with negative scores

    print("\nScore stats (dimensionless, should be in [-1, 0]):")
    print(f"  score_min = {scores.min():.4f}")
    print(f"  score_max = {scores.max():.4f}")
    print(f"  score_mean = {scores.mean():.4f}")

    print("\nReward stats (R = exp(-beta*score)):")
    print(f"  R_min = {rewards.min():.3e}")
    print(f"  R_max = {rewards.max():.3e}")
    print(f"  R_mean = {rewards.mean():.3e}")
    print("  log10(R) quantiles:")
    logR = np.log10(rewards)
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = np.quantile(logR, q)
        print(f"    q={q:4.2f} : log10(R) = {v:8.3f}")

    # 5) Print a small table of a few sample states
    print("\n=== Sample states (every ~10th sample) ===")
    header = (
        "idx   "
        + " ".join([f"θ{i}(rad)" for i in range(min(env.n_dim, 4))])
        + "   E(Hartree)      score      reward"
    )
    print(header)
    print("-" * len(header))

    for idx in range(0, n_samples, max(1, n_samples // 10)):
        st = states[idx]
        angle_str = " ".join([f"{a:7.3f}" for a in st[:4]])  # show first 4 angles
        print(
            f"{idx:3d}   {angle_str}   "
            f"{energies[idx]: .9f}   "
            f"{scores[idx]: 7.3f}   "
            f"{rewards[idx]: .3e}"
        )


if __name__ == "__main__":
    main()
