"""
Runnable script with hydra capabilities
"""

# This is a hotfix for tblite (used for the conformer generation) not
# importing correctly unless it is being imported first.
try:
    from tblite import interface
except:
    pass

import os
import pickle
import random
import sys

import hydra
import pandas as pd
import numpy as np
from pathlib import Path

from gflownet.utils.common import chdir_random_subdir
from gflownet.utils.policy import parse_policy_config


import numpy as np  # you already have this somewhere above, keep it

def parse_state_to_array(st):
    """
    Convert a replay-buffer 'state' entry to a 1D numpy array of floats.

    Handles:
      - list/tuple/np.ndarray
      - string like "[... ...] | 5"
    """
    # Already numeric
    if isinstance(st, (list, tuple, np.ndarray)):
        return np.asarray(st, dtype=float)

    # String representation: "[... ...] | t"
    if isinstance(st, str):
        # Drop the trailing "| t" part if present
        if "|" in st:
            coord_part = st.split("|", 1)[0].strip()
        else:
            coord_part = st.strip()

        # Remove brackets and split on whitespace
        coord_part = coord_part.replace("[", " ").replace("]", " ")
        tokens = coord_part.split()
        return np.asarray([float(x) for x in tokens], dtype=float)

    raise TypeError(f"Unsupported state type: {type(st)}")


@hydra.main(config_path="./config", config_name="main", version_base="1.1")
def main(config):
    # TODO: fix race condition in a more elegant way
    chdir_random_subdir()

    # Get current directory and set it as root log dir for Logger
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    print(f"\nLogging directory of this run:  {cwd}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    # The proxy is passed to env and used for computing rewards
    env = hydra.utils.instantiate(
        config.env,
        proxy=proxy,
        device=config.device,
        float_precision=config.float_precision,
    )

    # --- DEBUG: print internal DOFs ---
    print("\n[DEBUG] Conformer env from main.py")
    print("  SMILES:", getattr(env, "smiles", "<unknown>"))
    print("  n_dim (internal DOFs):", getattr(env, "n_dim", "<no n_dim>"))
    if hasattr(env, "n_torsion"):
        print("    -> torsions:     ", env.n_torsion)
    if hasattr(env, "n_bond_length"):
        print("    -> bond lengths: ", env.n_bond_length)
    if hasattr(env, "n_bond_angle"):
        print("    -> bond angles:  ", env.n_bond_angle)
    print()

    # The policy is used to model the probability of a forward/backward action
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")

    forward_policy = hydra.utils.instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = hydra.utils.instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )

    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        buffer=config.env.buffer,
        logger=logger,
    )
    gflownet.train()

    # Sample from trained GFlowNet
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = env.oracle(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        df.to_csv("gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open("gfn_samples.pkl", "wb"))
        # TODO: refactor before merging
        dct["conformer"] = [env.set_conformer(state).rdk_mol for state in x_sampled]
        pickle.dump(
            dct, open(f"conformers_{env.smiles}_{type(env.proxy).__name__}.pkl", "wb")
        )

        # Print replay buffer
    if len(gflownet.buffer.replay) > 0:
        print("\nReplay buffer:")
        print(gflownet.buffer.replay)

        # ===== Save replay buffer and top states for analysis =====
        run_dir = Path(cwd)

        rb = gflownet.buffer.replay.copy()
        # 1) full replay buffer
        rb.to_pickle(run_dir / "replay_buffer.pkl")

        # 2) top-K lowest-energy states
        rb_sorted = rb.sort_values("energy")  # lower energy = better
        K = min(200, len(rb_sorted))
        top = rb_sorted.head(K)

        # states: list of numpy arrays (ragged allowed)
        states_list = [parse_state_to_array(st) for st in top["state"]]

        # Try to save as a stacked array *only if* all have same length
        lengths = {s.shape[0] for s in states_list}
        if len(lengths) == 1:
            states_arr = np.stack(states_list)  # (K, state_dim)
            np.save(run_dir / "top_states.npy", states_arr)
        else:
            print(f"[WARN] top states have varying lengths {lengths}, "
                "skipping save of top_states.npy")

        # proxy states: (Z, x, y, z) per atom — this works fine even with ragged state length
        proxy_states = env.statebatch2proxy(states_list)  # (K, n_atoms, 4)

        np.save(run_dir / "top_proxy_states.npy", proxy_states)
        np.save(run_dir / "top_energies.npy", top["energy"].to_numpy())

        print(f"\n[INFO] Saved replay_buffer.pkl and top_* files in {run_dir}\n")

    # Close logger
    gflownet.logger.end()



def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
