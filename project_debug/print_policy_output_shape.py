from omegaconf import OmegaConf
from gflownet.envs.conformers.conformer import Conformer

# Minimal fake config just to instantiate the env
cfg = OmegaConf.create(
    {
        "length_traj": 16,
        "vonmises_mean": 0.0,
        "vonmises_concentration": 0.0,
        "n_comp": 1,
    }
)

env = Conformer(
    smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    n_torsion_angles=2,
    flex_bond_lengths=[],
    flex_bond_angles=[],
    length_traj=cfg.length_traj,
)

# The params dict ContinuousTorus expects
params = {
    "vonmises_mean": cfg.vonmises_mean,
    "vonmises_concentration": cfg.vonmises_concentration,
}

out = env.get_policy_output(params)
print("Type of get_policy_output:", type(out))
print("If it's a tuple, length:", len(out) if isinstance(out, tuple) else "n/a")
print("Value:", out)
