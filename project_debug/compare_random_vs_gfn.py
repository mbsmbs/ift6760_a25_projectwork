import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from gflownet.envs.conformers.conformer import Conformer
from gflownet.proxy.conformers.torchani import TorchANIMoleculeEnergy


def build_env_and_proxy(device="cpu"):
    """
    Rebuild the Conformer env and TorchANI proxy roughly like in mlp_torchani.yaml.
    """
    env = Conformer(
        smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        n_torsion_angles=-1,
        length_traj=5,
        policy_type="mlp",
    )

    proxy = TorchANIMoleculeEnergy(
        model="ANI2x",
        use_ensemble=True,
        batch_size=128,
        n_samples=256,
        normalize=True,
        # >>> these two kwargs were missing <<<
        float_precision="float32",
        device=device,
    )

    proxy.setup(env)
    return env, proxy



def sample_random_normalized_energies(env, proxy, n_samples, seed=0):
    """
    Uniformly sample internal coordinates in [0, 2*pi]^n_dim,
    append time=1, convert to proxy states, and evaluate normalized energies.
    """
    rng = np.random.default_rng(seed)

    # Internal DOFs
    internal = 2 * np.pi * rng.random((n_samples, env.n_dim))

    # Time coordinate = 1
    t_col = np.ones((n_samples, 1), dtype=float)
    env_states = np.concatenate([internal, t_col], axis=1)

    # Convert to proxy format (atoms + positions)
    proxy_states = env.statebatch2proxy(env_states)

    # Compute normalized energies using MoleculeEnergyBase.__call__
    with torch.no_grad():
        energies = proxy(proxy_states).cpu().numpy()

    return energies


def main():
    # === 1. Load GFlowNet samples ===
    logdir = (
        "logs/hybrid_extended/debug_reward_03/847e74f9"
    )  # <- adjust if you run a new experiment
    csv_path = os.path.join(logdir, "gfn_samples.csv")

    df = pd.read_csv(csv_path)
    if "energies" not in df.columns:
        raise RuntimeError(f"'energies' column not found in {csv_path}")

    energies_gfn = df["energies"].values
    n_gfn = len(energies_gfn)

    print("=== GFlowNet samples (normalized energies) ===")
    print(f"n:   {n_gfn}")
    print(f"min: {energies_gfn.min(): .6f}")
    print(f"max: {energies_gfn.max(): .6f}")
    print(f"mean:{energies_gfn.mean(): .6f}")
    print(f"std: {energies_gfn.std(): .6f}")
    print()

    # === 2. Build env + proxy and sample random states ===
    env, proxy = build_env_and_proxy(device="cpu")

    energies_rand = sample_random_normalized_energies(env, proxy, n_gfn, seed=123)
    print("=== Random internal states (normalized energies) ===")
    print(f"n:   {len(energies_rand)}")
    print(f"min: {energies_rand.min(): .6f}")
    print(f"max: {energies_rand.max(): .6f}")
    print(f"mean:{energies_rand.mean(): .6f}")
    print(f"std: {energies_rand.std(): .6f}")

    # === 3. Plot comparison ===
    plt.figure(figsize=(7, 5))
    bins = np.linspace(-1.0, 0.0, 35)

    plt.hist(
        energies_rand,
        bins=bins,
        alpha=0.5,
        density=True,
        label="Random internal states",
    )
    plt.hist(
        energies_gfn,
        bins=bins,
        alpha=0.5,
        density=True,
        label="GFlowNet samples",
    )
    plt.xlabel("normalized energy")
    plt.ylabel("density")
    plt.title("Energy distribution: random vs GFlowNet samples")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(logdir, "random_vs_gfn_energies.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved comparison plot to: {out_png}")


if __name__ == "__main__":
    main()
