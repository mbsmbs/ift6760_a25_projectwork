# inspect_env.py
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate


def main():
    # This file is at repo root
    root = Path(__file__).resolve().parent
    print("Here:", root)

    # Paths adapted to YOUR tree:
    exp_path = root / "config" / "experiments" / "ai4mat23" / "mlp_torchani.yaml"
    env_base_path = root / "config" / "env" / "conformers" / "conformer.yaml"

    print("Experiment config:", exp_path)
    print("Base env config:", env_base_path)

    # 1) Load configs
    exp_cfg = OmegaConf.load(exp_path)
    env_base_cfg = OmegaConf.load(env_base_path)

    # 2) Merge base env + experiment overrides (env section of mlp_torchani)
    env_overrides = exp_cfg.env
    env_cfg = OmegaConf.merge(env_base_cfg, env_overrides)

    print("\n=== FINAL ENV CONFIG (after merge) ===")
    print(OmegaConf.to_yaml(env_cfg))

    # 3) Instantiate the environment
    env = instantiate(env_cfg)
    print("\nEnv type:", type(env))

    # 4) Inspect a few attributes if they exist
    for attr in ["n_torsion_angles", "n_dim", "smiles", "reward_beta"]:
        if hasattr(env, attr):
            print(f"{attr}:", getattr(env, attr))

    # 5) Try to get an initial state (optional)
    if hasattr(env, "get_init_state"):
        try:
            s0 = env.get_init_state(batch_size=1)
            print("\nInitial state shape:", getattr(s0, "shape", None))
            print("Initial state example:", s0[0] if hasattr(s0, "__getitem__") else s0)
        except Exception as e:
            print("\n(get_init_state) raised an error:", repr(e))


if __name__ == "__main__":
    main()
