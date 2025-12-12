import os
import numpy as np
import pandas as pd

from rdkit import Chem

from gflownet.envs.conformers.conformer import Conformer

# === CONFIG ===
# Directory of the run you want to inspect
RUN_DIR = "logs/hybrid_extended/debug_reward_03/847e74f9"
GFN_CSV = os.path.join(RUN_DIR, "gfn_samples.csv")

# SMILES of the molecule used in the run
SMILES = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"


def parse_state_str(s: str) -> np.ndarray:
    """
    Robustly parse the 'readable' column from gfn_samples.csv into a numpy array.

    Handles things like:
      "[304.17465093  22.85101259 ... 202.44380536]"
      with extra '|' or newlines.
    """
    s = str(s)
    s = s.replace("[", " ").replace("]", " ").replace("|", " ")
    s = s.replace("\n", " ")

    arr = np.fromstring(s, sep=" ")
    if arr.size == 0:
        raise ValueError(f"Failed to parse state string: {s[:120]}...")
    return arr


def build_torsion_only_env() -> Conformer:
    """
    Build a Conformer env *without* extra bond-length/angle DOFs.
    This gives a torsion-only environment with env.n_dim = #torsions.
    """
    env = Conformer(
        smiles=SMILES,
        n_torsion_angles=-1,   # discover all torsions
        policy_type="mlp",
    )
    print(f"[INFO] Built torsion-only env with env.n_dim = {env.n_dim}")
    return env


def main():
    # -----------------------------
    # 1) Load GFlowNet samples
    # -----------------------------
    if not os.path.exists(GFN_CSV):
        raise FileNotFoundError(f"Could not find gfn_samples.csv at {GFN_CSV}")

    df = pd.read_csv(GFN_CSV)
    print("Columns in gfn_samples.csv:", df.columns.tolist())

    states_mat = np.stack(df["readable"].apply(parse_state_str).values)
    print("Parsed GFN states shape:", states_mat.shape)  # (N, 25) = 24 internal + time

    # Separate internal coords and time
    internal_all = states_mat[:, :-1]   # drop time dim from file (last column)
    internal_dim_from_file = internal_all.shape[1]
    print(f"[INFO] internal_dim_from_file = {internal_dim_from_file}")

    # -----------------------------
    # 2) Build torsion-only env
    # -----------------------------
    env = build_torsion_only_env()
    env_dim = env.n_dim

    if internal_dim_from_file < env_dim:
        raise ValueError(
            f"File internal dim {internal_dim_from_file} < env.n_dim {env_dim}, "
            "cannot map states safely."
        )

    if internal_dim_from_file > env_dim:
        print(
            f"[WARN] internal_dim_from_file={internal_dim_from_file}, "
            f"env.n_dim={env_dim}. Using only first {env_dim} coordinates "
            "from each GFN state (torsion-only view)."
        )
        internal = internal_all[:, :env_dim]
    else:
        internal = internal_all

    # Re-append a dummy time coordinate (1.0) for the env state format
    gfn_states_for_env = np.concatenate(
        [internal, np.ones((internal.shape[0], 1))], axis=1
    )
    print(f"[INFO] gfn_states_for_env shape: {gfn_states_for_env.shape}")

    # -----------------------------
    # 3) Select lowest-energy GFN states
    # -----------------------------
    if "energies" not in df.columns:
        raise KeyError("Column 'energies' not found in gfn_samples.csv")

    energies_norm = df["energies"].values
    order = np.argsort(energies_norm)  # ascending (more negative is better)
    n_gfn = min(50, len(order))

    # -----------------------------
    # 4) Write SDF for GFN low-energy conformers
    # -----------------------------
    gfn_sdf_path = os.path.join(RUN_DIR, "gfn_lowE_conformers_torsion_only.sdf")
    writer_gfn = Chem.SDWriter(gfn_sdf_path)

    print(f"[INFO] Writing {n_gfn} lowest-energy GFN conformers to {gfn_sdf_path}")

    for idx in order[:n_gfn]:
        st = gfn_states_for_env[idx]
        conf = env.sync_conformer_with_state(st)

        # Copy RDKit mol so we don't share mutable state
        mol = Chem.Mol(conf.rdk_mol)
        # Attach the (normalized) energy as an SD property
        mol.SetProp("gfn_energy_norm", f"{energies_norm[idx]:.6f}")
        writer_gfn.write(mol)

    writer_gfn.close()

    # -----------------------------
    # 5) Sample random torsion states + write SDF
    # -----------------------------
    rng = np.random.default_rng(0)
    rand_internal = 2 * np.pi * rng.random((n_gfn, env_dim))
    rand_states_for_env = np.concatenate(
        [rand_internal, np.ones((n_gfn, 1))], axis=1
    )

    rand_sdf_path = os.path.join(RUN_DIR, "random_conformers_torsion_only.sdf")
    writer_rand = Chem.SDWriter(rand_sdf_path)

    print(f"[INFO] Writing {n_gfn} random torsion-only conformers to {rand_sdf_path}")

    for i in range(n_gfn):
        st = rand_states_for_env[i]
        conf = env.sync_conformer_with_state(st)
        mol = Chem.Mol(conf.rdk_mol)
        writer_rand.write(mol)

    writer_rand.close()

    print("[DONE] Wrote SDF files:")
    print("  -", gfn_sdf_path)
    print("  -", rand_sdf_path)


if __name__ == "__main__":
    main()
