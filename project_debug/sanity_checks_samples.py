# project_debug/sanity_checks_samples.py

import pandas as pd
import pickle
from pathlib import Path

# Repo root = parent of project_debug
ROOT = Path(__file__).resolve().parents[1]

# Folder where Hydra saved your sampled run
runs_root = ROOT / "logs" / "local_run_samples"

# keep only subdirectories (ignore main.log, etc.)
run_dirs = sorted([d for d in runs_root.iterdir() if d.is_dir()])
if not run_dirs:
    raise RuntimeError(
        f"No run directories found in {runs_root}. "
        "Did you already run main.py with hydra.run.dir=./logs/local_run_samples ?"
    )

run_dir = run_dirs[-1]  # latest run directory
print(f"Using run directory: {run_dir}")


# 1) Check CSV summary
df = pd.read_csv(run_dir / "gfn_samples.csv")
print(df.head())
print("Energy stats:\n", df["energies"].describe())

# 2) Load conformers (RDKit objects)
pkl_name = "conformers_CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O_TorchANIMoleculeEnergy.pkl"
pkl_path = run_dir / pkl_name

if not pkl_path.exists():
    raise FileNotFoundError(f"Could not find {pkl_path}")

dct = pickle.load(open(pkl_path, "rb"))
print("Number of conformers:", len(dct["conformer"]))
