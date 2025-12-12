# project_debug/compare_rmsd_and_energy.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import rdMolAlign

# ---------------------------------------------------------------------
# UPDATE THESE TWO IDs IF YOU RERUN TRAINING
# ---------------------------------------------------------------------
BASE_RUN = "ac58ff16"   # torsions only
EXT_RUN  = "25e22337"   # torsions + bond lengths + angles

base_dir = f"logs/hybrid_extended/exp_base_torsion/{BASE_RUN}"
ext_dir  = f"logs/hybrid_extended/exp_extBLA_big/{EXT_RUN}"


def load_conformers_and_energies(run_dir):
    """
    Loads RDKit conformers and the corresponding energies.

    We use the conformers_*.pkl file written by main.py:
        dct = {"x": x_sampled, "energy": energies, "conformer": [...]}
    """
    pkls = [f for f in os.listdir(run_dir) if f.startswith("conformers_") and f.endswith(".pkl")]
    if not pkls:
        raise FileNotFoundError(f"No conformers_*.pkl file found in {run_dir}")
    path = os.path.join(run_dir, pkls[0])
    dct = pickle.load(open(path, "rb"))

    mols = dct["conformer"]          # list of RDKit Mol
    energies = np.asarray(dct["energy"], dtype=float)  # shape (K,)
    return mols, energies


def compute_rmsd_to_best(mols, energies):
    """
    Compute RMSD of each conformer to the *best-energy* conformer.

    We assume each mol has a single conformer (confId=0).
    """
    best_idx = np.argmin(energies)
    ref = mols[best_idx]
    rmsds = []
    for m in mols:
        rms = rdMolAlign.AlignMol(prbMol=m, refMol=ref, prbCid=0, refCid=0)
        rmsds.append(rms)
    return np.asarray(rmsds, dtype=float)


def normalize_energies(e_base, e_ext):
    """
    Normalize energies to something like the slide:
    subtract global min so the best conformer has energy 0.
    """
    global_min = min(e_base.min(), e_ext.min())
    e_base_norm = e_base - global_min
    e_ext_norm = e_ext - global_min
    return e_base_norm, e_ext_norm

def summary_stats(name, rmsd, e_norm):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"# samples                 : {len(rmsd)}")
    print(f"RMSD  min / mean / max    : "
          f"{rmsd.min():.3f} / {rmsd.mean():.3f} / {rmsd.max():.3f} Å")
    print(f"Energy(min-shifted) min / mean / max : "
          f"{e_norm.min():.3f} / {e_norm.mean():.3f} / {e_norm.max():.3f}")


if __name__ == "__main__":
    print("Base dir:", base_dir)
    print("Ext  dir:", ext_dir)

    # -----------------------------------------------------------------
    # 1) Load conformers & energies for both runs
    # -----------------------------------------------------------------
    mols_base, E_base = load_conformers_and_energies(base_dir)
    mols_ext,  E_ext  = load_conformers_and_energies(ext_dir)

    # -----------------------------------------------------------------
    # 2) Compute RMSDs to best-energy conformer in each run
    # -----------------------------------------------------------------
    rmsd_base = compute_rmsd_to_best(mols_base, E_base)
    rmsd_ext  = compute_rmsd_to_best(mols_ext,  E_ext)

    # -----------------------------------------------------------------
    # 3) Normalize energies (for nicer axis)
    # -----------------------------------------------------------------
    E_base_norm, E_ext_norm = normalize_energies(E_base, E_ext)

    # ----- Print numeric summary (for the report) -----
    summary_stats("Torsions only", rmsd_base, E_base_norm)
    summary_stats("Extended (torsions + BLA)", rmsd_ext, E_ext_norm)

    # -----------------------------------------------------------------
    # 4) Make the 2-panel figure (similar to the slide)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ---- Left: Energy vs RMSD ----
    ax0 = axes[0]
    ax0.scatter(rmsd_ext,  E_ext_norm,  s=20, alpha=0.7, label="Extended (torsions + BLA)")
    ax0.scatter(rmsd_base, E_base_norm, s=20, alpha=0.7, label="Torsions only")
    ax0.set_xlabel("RMSD to best-energy conformer (Å)")
    ax0.set_ylabel("TorchANI energy (normalized)")
    ax0.set_title("Energy vs RMSD of sampled conformers")
    ax0.legend()

    # ---- Right: Energy histogram ----
    ax1 = axes[1]
    all_E = np.concatenate([E_base_norm, E_ext_norm])
    bins = np.linspace(all_E.min(), all_E.max(), 30)

    ax1.hist(E_ext_norm,  bins=bins, alpha=0.7, label="Extended (torsions + BLA)")
    ax1.hist(E_base_norm, bins=bins, alpha=0.7, label="Torsions only")
    ax1.set_xlabel("TorchANI energy (normalized)")
    ax1.set_ylabel("Count")
    ax1.set_title("Energy distribution of sampled conformers")
    ax1.legend()

    fig.suptitle("Comparison (extended vs torsions-only)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = "energy_rmsd_comparison.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")
