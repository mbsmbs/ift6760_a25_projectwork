import os
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def main():
    # Absolute path to the config folder
    here = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(here, "config")
    print("Here:", here)
    print("Using config_dir:", config_dir)

    with initialize_config_dir(
        config_dir=config_dir,
        job_name="debug",
        version_base=None,  # for Hydra >=1.1
    ):
        # Use the SAME base config as main.py: config/config.yaml
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiments=ai4mat23/mlp_torchani",          # <--- NOTE THE +
                "device=cpu",
                'env.smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"',
                "proxy=conformers/torchani",
                "logger.do.online=False",
            ],
        )

        print("\n=== ENV CONFIG FROM HYDRA ===")
        print(OmegaConf.to_yaml(cfg.env))

        # Instantiate environment
        env = instantiate(cfg.env, _recursive_=False)
        print("\nEnv type:", type(env))

        # Some basic probes
        if hasattr(env, "n_dim"):
            print("n_dim (number of internal DOFs):", env.n_dim)

        # Sample initial internal coordinates
        if hasattr(env, "sample_initial_states"):
            states = env.sample_initial_states(n=2)
            print("\nInitial states:")
            print("  type:", type(states))
            if hasattr(states, "shape"):
                print("  shape:", getattr(states, "shape", None))
            try:
                print("  first state:", states[0])
            except Exception as e:
                print("  could not index first state:", e)
        else:
            print("env has no method sample_initial_states")

        # Energies and rewards
        if hasattr(env, "energy"):
            energies = env.energy(states)
            print("\nEnergies:", energies)
        if hasattr(env, "reward"):
            rewards = env.reward(states)
            print("Rewards:", rewards)


if __name__ == "__main__":
    main()
