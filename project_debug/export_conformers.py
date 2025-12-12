import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ✅ New run IDs
BASE_RUN = "ac58ff16"
EXT_RUN  = "25e22337"

base_dir = f"logs/hybrid_extended/exp_base_torsion/{BASE_RUN}"
ext_dir  = f"logs/hybrid_extended/exp_extBLA_big/{EXT_RUN}"

smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"


def write_sdf(proxy_path, out_prefix, num_confs=10):
    proxy = np.load(proxy_path)  # (K, n_atoms, 4)

    for idx in range(min(num_confs, proxy.shape[0])):
        arr = proxy[idx]  # (n_atoms, 4)
        Z = arr[:, 0].astype(int)
        coords = arr[:, 1:4]

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer()

        # sanity check
        assert mol.GetNumAtoms() == coords.shape[0]

        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))

        w = Chem.SDWriter(f"{out_prefix}_{idx}.sdf")
        w.write(mol)
        w.close()


if __name__ == "__main__":
    # ✅ Use the base_dir / ext_dir defined above
    print("Base dir:", base_dir)
    print("Ext dir :", ext_dir)

    write_sdf(f"{base_dir}/top_proxy_states.npy",
              f"{base_dir}/torsion_only_conf")
    write_sdf(f"{ext_dir}/top_proxy_states.npy",
              f"{ext_dir}/extBLA_conf")
