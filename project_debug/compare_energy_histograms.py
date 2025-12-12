import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# --- 1) Locate the two runs ---

root = Path("/Users/byungsukmin/Desktop/udem/INF/IFT6760/Project/ift6760_a25_projectwork")

extended_root = root / "logs" / "local_run_samples"
torsion_root  = root / "logs" / "local_run_torsions_only"

# pick latest subdir in each
def latest_run(run_root: Path) -> Path:
    subdirs = [p for p in run_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No run directories found in {run_root}")
    return sorted(subdirs)[-1]

extended_dir = latest_run(extended_root)
torsion_dir  = latest_run(torsion_root)

print("[INFO] Extended env run dir:", extended_dir)
print("[INFO] Torsions-only run dir:", torsion_dir)

# --- 2) Load CSVs with sampled energies ---

df_ext = pd.read_csv(extended_dir / "gfn_samples.csv")
df_tor = pd.read_csv(torsion_dir / "gfn_samples.csv")

print("\n[Extended env] energy stats:")
print(df_ext["energies"].describe())

print("\n[Torsions only] energy stats:")
print(df_tor["energies"].describe())

# --- 3) Plot histograms ---

plt.figure(figsize=(6, 4))
plt.hist(df_ext["energies"], bins=20, alpha=0.5, label="Extended (torsions + bonds + angles)")
plt.hist(df_tor["energies"], bins=20, alpha=0.5, label="Torsions only")
plt.xlabel("TorchANI energy (normalized units from env)")
plt.ylabel("Count")
plt.title("Energy distribution of sampled conformers")
plt.legend()
plt.tight_layout()
plt.show()
