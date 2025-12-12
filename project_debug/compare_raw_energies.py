import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from gflownet.envs.conformers.conformer import Conformer
from gflownet.proxy.conformers.torchani import TorchANIMoleculeEnergy


# ====== CONFIG ======
# Path to the run directory where gfn_samples.csv lives
RUN_DIR = "logs/hybrid_extended/debug_reward_03/847e74f9"
CSV_PATH = os.path.join(RUN_DIR, "gfn_samples.csv")
OUT_PLOT = os.path.join(RUN_DIR, "raw_random_vs_gfn_energies.png")

DEVICE = "cpu"

# ====== Robust parser for the 'readable' column ======

# Match floats like:
#  1, -2.3, 4.5e-3, -1.2E+4, etc.
float_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_state_str(s: str) -> np.ndarray:
    """
    Parse a 'readable' string like:
        "[304.17  22.85 ... 202.44]"
    possibly with junk characters, into a 1D numpy array of floats.
    Non-numeric junk (like '|' or brackets) is ignored.
    """
    s = str(s)
    matches = float_pattern.findall(s)
    vals = [float(x) for x in matches]
    return np.array(vals, dtype=float)


# ====== Build env + proxy manually (no Hydra config.yaml needed) ======

def build_env_and_proxy(device: str = "cpu"):
    """
    Build a *torsion-only* Conformer env + TorchANI proxy.

    NOTE: This env has env.n_dim = n_torsion_angles (e.g. 8 for ibuprofen).
    The GFN states from training may have more internal DOFs (due to bond
    lengths/angles), but for this debug script we will *truncate* them to
    env.n_dim.
    """
    env = Conformer(
        smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        n_torsion_angles=-1,     # use all rotatable torsions
        torsion_indices=None,
        policy_type="mlp",
        remove_hs=True,
    )

    print("DEBUG env.n_dim:", env.n_dim)

    proxy = TorchANIMoleculeEnergy(
        model="ANI2x",
        use_ensemble=True,
        batch_size=128,
        n_samples=256,
        normalize=False,          # get RAW TorchANI energies
        clamp=False,
        remove_outliers=False,
        float_precision="float32",
        device=device,
    )

    # Estimate min/max on random internal states (for completeness; not used for reward)
    proxy.setup(env)

    return env, proxy


def main():
    RUN_DIR = "logs/hybrid_extended/debug_reward_03/847e74f9"
    csv_path = f"{RUN_DIR}/gfn_samples.csv"

    print(f"Loading replay buffer from: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- 1) Parse GFlowNet states (internal DOFs + time) ---
    states_gfn = np.stack(df["readable"].apply(parse_state_str).values)
    print("GFlowNet states shape:", states_gfn.shape)   # (N, D_file)

    N, D_file = states_gfn.shape
    # Assume last coordinate is time
    internal_dim_file = D_file - 1
    states_gfn_internal = states_gfn[:, :internal_dim_file]

    # --- 2) Build env & proxy ---
    device = "cpu"
    env, proxy = build_env_and_proxy(device=device)

    # --- 3) Adjust for dimension mismatch (truncate if needed) ---
    if internal_dim_file != env.n_dim:
        print(
            f"[WARN] internal_dim_from_file={internal_dim_file}, "
            f"env.n_dim={env.n_dim}. "
            f"Will only use first {env.n_dim} coordinates from GFN states."
        )
        states_gfn_internal = states_gfn_internal[:, :env.n_dim]

    # --- 4) Compute RAW TorchANI energies for GFlowNet samples ---
    proxy_states_gfn = env.statebatch2proxy(states_gfn_internal)
    energies_gfn_raw = proxy.compute_energy(proxy_states_gfn).cpu().numpy()

    print("\n=== RAW TorchANI energies for GFlowNet samples ===")
    print(f"n:   {energies_gfn_raw.shape[0]:3d}")
    print(f"min: {energies_gfn_raw.min(): .6f}")
    print(f"max: {energies_gfn_raw.max(): .6f}")
    print(f"mean:{energies_gfn_raw.mean(): .6f}")
    print(f"std: {energies_gfn_raw.std(): .6f}")

    # --- 5) Random internal states baseline (same env.n_dim) ---
    n_random = N
    random_internal = 2 * np.pi * np.random.rand(n_random, env.n_dim)
    proxy_states_rand = env.statebatch2proxy(random_internal)
    energies_rand_raw = proxy.compute_energy(proxy_states_rand).cpu().numpy()

    print("\n=== RAW TorchANI energies for random internal states ===")
    print(f"n:   {energies_rand_raw.shape[0]:3d}")
    print(f"min: {energies_rand_raw.min(): .6f}")
    print(f"max: {energies_rand_raw.max(): .6f}")
    print(f"mean:{energies_rand_raw.mean(): .6f}")
    print(f"std: {energies_rand_raw.std(): .6f}")

    # Optional: save a small comparison plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(energies_rand_raw, bins=40, density=True, alpha=0.5, label="Random")
    plt.hist(energies_gfn_raw, bins=40, density=True, alpha=0.5, label="GFN")
    plt.xlabel("TorchANI energy (Hartree)")
    plt.ylabel("Density")
    plt.legend()
    out_path = f"{RUN_DIR}/raw_random_vs_gfn_energies_torsion_only.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nSaved torsion-only raw energy comparison to: {out_path}")


if __name__ == "__main__":
    main()
