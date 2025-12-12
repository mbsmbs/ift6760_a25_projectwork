# project_debug/dump_env_dofs.py
import hydra
from omegaconf import DictConfig
from gflownet.envs.conformers.conformer import Conformer

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # We mimic main.py: load the global config, then use CLI overrides.
    # So you should run this script with +experiments=... on the command line.
    print("Selected experiments:", cfg.get("experiments"))

    # Instantiate the env exactly like in training
    env: Conformer = hydra.utils.instantiate(cfg.env)

    print("SMILES:", env.smiles)
    print("n_dim (internal DOFs):", env.n_dim)
    print("n_torsion_angles:", env.n_torsion_angles)
    print("n_bond_lengths:", env.n_bond_lengths)
    print("n_bond_angles:", env.n_bond_angles)
    print("flex_bond_lengths:", env.flex_bond_lengths)
    print("flex_bond_angles:", env.flex_bond_angles)

if __name__ == "__main__":
    main()
