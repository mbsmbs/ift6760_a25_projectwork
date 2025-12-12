# ani_energy_debug.py
import os

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torchani
from rdkit import Chem
from rdkit.Chem import AllChem


def load_env_cfg():
    """Load and merge base conformer env config + experiment env overrides."""
    cwd = os.getcwd()
    exp_cfg_path = os.path.join(
        cwd, "config", "experiments", "ai4mat23", "mlp_torchani.yaml"
    )
    base_env_path = os.path.join(
        cwd, "config", "env", "conformers", "conformer.yaml"
    )

    exp_cfg = OmegaConf.load(exp_cfg_path)
    base_env_cfg = OmegaConf.load(base_env_path)

    # Merge base conformer env + experiment's env overrides
    env_cfg = OmegaConf.merge(base_env_cfg, exp_cfg.env)

    print(f"Here: {cwd}")
    print(f"Experiment config: {exp_cfg_path}")
    print(f"Base env config:   {base_env_path}\n")

    print("=== FINAL ENV CONFIG ===")
    print(OmegaConf.to_yaml(env_cfg))
    print()

    return env_cfg


def build_env(env_cfg):
    """Instantiate the Conformer env from the merged config."""
    env = instantiate(env_cfg)
    print("=== Conformer env ===")
    print(f"Type: {type(env)}")
    print(f"SMILES: {env.smiles}")
    print(f"n_dim (torsion DOFs): {env.n_dim}\n")
    return env


def smiles_to_atomic_numbers(smiles: str):
    """
    Build an RDKit molecule with explicit H and return atomic numbers.

    We deliberately go through RDKit here instead of using env.conformer.rdk_mol,
    so we don't depend on RDKitConformer internal attributes (which caused
    the previous 'no attribute rdk_mol' error).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return atomic_numbers


def conformer_to_ani_inputs(env):
    """
    Convert the current env.conformer into TorchANI (species, coordinates).

    - Coordinates: use env.conformer.get_atom_positions() (already working in your
      single_step_debug).
    - Species: recomputed from SMILES with RDKit + AddHs, using atomic numbers.
    """
    # Current 3D coordinates from env's RDKitConformer
    pos = env.conformer.get_atom_positions()  # numpy array (N, 3)

    # Atomic numbers via fresh RDKit molecule from the same SMILES
    Z_list = smiles_to_atomic_numbers(env.smiles)

    if len(Z_list) != pos.shape[0]:
        raise RuntimeError(
            f"Length mismatch between species ({len(Z_list)}) and positions "
            f"({pos.shape[0]}). This likely means RDKit atom ordering for the "
            "SMILES reconstruction does not match the conformer."
        )

    species = torch.tensor(Z_list, dtype=torch.long).unsqueeze(0)      # (1, N)
    coordinates = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)

    return species, coordinates


def main():
    # 1) Load env config and instantiate Conformer
    env_cfg = load_env_cfg()
    env = build_env(env_cfg)

    # 2) Device & TorchANI model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("/opt/anaconda3/envs/confgfn/lib/python3.8/site-packages/torchani/resources/")

    # Important: periodic_table_index=True so species tensor is atomic numbers
    model = torchani.models.ANI2x(periodic_table_index=True).to(device).eval()

    # 3) Build ANI inputs from current env.conformer
    species0, R0 = conformer_to_ani_inputs(env)
    species0 = species0.to(device)
    R0 = R0.to(device)

    # 4) Evaluate energy
    with torch.no_grad():
        out0 = model((species0, R0))
        # TorchANI returns an object with `.energies`
        E0 = out0.energies.item()

    print("\n=== TorchANI energy for current env.conformer ===")
    print(f"Species (first 10): {species0[0, :10].tolist()} ...")
    print(f"Coordinates shape: {tuple(R0.shape)}")
    print(f"Energy (Hartree): {E0:.6f}")


if __name__ == "__main__":
    main()
