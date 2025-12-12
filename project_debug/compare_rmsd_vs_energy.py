from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

root = Path("/Users/byungsukmin/Desktop/udem/INF/IFT6760/Project/ift6760_a25_projectwork")

ext_dir = root / "logs" / "local_run_samples" / "806940cf"
tor_dir = root / "logs" / "local_run_torsions_only" / "c28d45e2"

df_ext = pd.read_csv(ext_dir / "gfn_samples.csv")
df_tor = pd.read_csv(tor_dir / "gfn_samples.csv")

pkl_ext = [p for p in ext_dir.glob("conformers_*TorchANIMoleculeEnergy.pkl")][0]
pkl_tor = [p for p in tor_dir.glob("conformers_*TorchANIMoleculeEnergy.pkl")][0]

d_ext = pickle.load(open(pkl_ext, "rb"))
d_tor = pickle.load(open(pkl_tor, "rb"))

confs_ext = d_ext["conformer"]   # list of RDKit Mol
confs_tor = d_tor["conformer"]

def rmsd_to_best(confs, energies):
    # index of lowest-energy conformer
    best_idx = energies.idxmin()
    ref = confs[best_idx]
    ref_id = ref.GetConformer().GetId()

    rmsds = []
    for m in confs:
        cid = m.GetConformer().GetId()
        rms = AllChem.GetBestRMS(ref, m, refId=ref_id, prbId=cid)
        rmsds.append(rms)
    return best_idx, rmsds

best_ext, rmsd_ext = rmsd_to_best(confs_ext, df_ext["energies"])
best_tor, rmsd_tor = rmsd_to_best(confs_tor, df_tor["energies"])

df_ext["rmsd_to_best"] = rmsd_ext
df_tor["rmsd_to_best"] = rmsd_tor

print("\n[Extended env] RMSD stats:")
print(df_ext["rmsd_to_best"].describe())

print("\n[Torsions only] RMSD stats:")
print(df_tor["rmsd_to_best"].describe())

# Scatter: energy vs RMSD
plt.figure(figsize=(6,4))
plt.scatter(df_ext["rmsd_to_best"], df_ext["energies"], alpha=0.7, label="Extended")
plt.scatter(df_tor["rmsd_to_best"], df_tor["energies"], alpha=0.7, label="Torsions only")
plt.xlabel("RMSD to best-energy conformer (Å)")
plt.ylabel("TorchANI energy (normalized)")
plt.title("Energy vs RMSD of sampled conformers")
plt.legend()
plt.tight_layout()
plt.show()
