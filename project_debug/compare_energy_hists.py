# project_debug/compare_energy_hists.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === update these if you rerun experiments ===
BASE_RUN = "ac58ff16"   # torsion-only run
EXT_RUN  = "25e22337"   # extended BLA run

base_dir = f"logs/hybrid_extended/exp_base_torsion/{BASE_RUN}"
ext_dir  = f"logs/hybrid_extended/exp_extBLA_big/{EXT_RUN}"

# ---- load replay buffers ----
rb_base = pd.read_pickle(f"{base_dir}/replay_buffer.pkl")
rb_ext  = pd.read_pickle(f"{ext_dir}/replay_buffer.pkl")

E_base = rb_base["energy"].to_numpy()
E_ext  = rb_ext["energy"].to_numpy()

print("torsions only:    ", E_base.min(), E_base.max())
print("torsions + BLA:   ", E_ext.min(), E_ext.max())

# ---- build bins ADAPTED to the data range ----
emin = min(E_base.min(), E_ext.min())
emax = max(E_base.max(), E_ext.max())

# tiny padding so the outer bins are not empty
delta = emax - emin if emax > emin else 1e-6
emin -= 0.05 * delta
emax += 0.05 * delta

bins = np.linspace(emin, emax, 50)

plt.figure(figsize=(8, 4))
plt.hist(E_base, bins=bins, alpha=0.5, label="torsions only", density=False)
plt.hist(E_ext,  bins=bins, alpha=0.5, label="torsions + BLA", density=False)

plt.xlabel("GFlowNet score (TorchANI-based)")
plt.ylabel("Count")
plt.title("Energy distributions: torsion-only vs extended BLA\n(replay buffer states)")
plt.legend()

plt.tight_layout()
plt.savefig("energy_hist_comparison_zoomed.png", dpi=200)
plt.close()
print("Saved energy_hist_comparison_zoomed.png")
