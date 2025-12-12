import os
from typing import List

import torch
import torchani
from rdkit import Chem

from gflownet.proxy.conformers.torchani import TORCHANI_MODELS

RUN_DIR = "logs/hybrid_extended/debug_reward_03/847e74f9"
GFN_SDF   = os.path.join(RUN_DIR, "gfn_lowE_conformers_torsion_only.sdf")
RAND_SDF  = os.path.join(RUN_DIR, "random_conformers_torsion_only.sdf")

MODEL_NAME = "ANI2x"
DEVICE = "cpu"


def load_mols_from_sdf(path: str) -> List[Chem.Mol]:
    suppl = Chem.SDMolSupplier(path, removeHs=False)
    mols = [m for m in suppl if m is not None]
    if len(mols) == 0:
        raise ValueError(f"No valid molecules found in {path}")
    return mols


def mols_to_torchani_batch(mols: List[Chem.Mol], device: str = "cpu"):
    atomic_nums = []
    coords = []

    for mol in mols:
        conf = mol.GetConformer()
        n = mol.GetNumAtoms()

        zs = []
        xyz = []
        for i in range(n):
            atom = mol.GetAtomWithIdx(i)
            zs.append(atom.GetAtomicNum())
            pos = conf.GetAtomPosition(i)
            xyz.append([pos.x, pos.y, pos.z])

        atomic_nums.append(torch.tensor(zs, dtype=torch.long))
        coords.append(torch.tensor(xyz, dtype=torch.float))

    # torchani expects (batch, atoms) and (batch, atoms, 3)
    atomic_nums = torch.nn.utils.rnn.pad_sequence(
        atomic_nums, batch_first=True, padding_value=0
    )
    coords = torch.nn.utils.rnn.pad_sequence(
        coords, batch_first=True, padding_value=0.0
    )

    return atomic_nums.to(device), coords.to(device)


def score_sdf(path: str):
    print(f"\n=== Scoring conformers from: {path} ===")
    mols = load_mols_from_sdf(path)
    print(f"Loaded {len(mols)} molecules.")

    elements, coordinates = mols_to_torchani_batch(mols, device=DEVICE)

    model = TORCHANI_MODELS[MODEL_NAME](
        periodic_table_index=True, model_index=None
    ).to(DEVICE)
    with torch.no_grad():
        energies = model((elements, coordinates)).energies.cpu().numpy().reshape(-1)

    print(f"n:   {len(energies)}")
    print(f"min: {energies.min(): .6f}")
    print(f"max: {energies.max(): .6f}")
    print(f"mean:{energies.mean(): .6f}")
    print(f"std: {energies.std(): .6f}")


def main():
    score_sdf(GFN_SDF)
    score_sdf(RAND_SDF)


if __name__ == "__main__":
    main()
