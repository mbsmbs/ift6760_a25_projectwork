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
import numpy as np
import pandas as pd

from gflownet.utils.common import chdir_random_subdir
from gflownet.utils.policy import parse_policy_config
from omegaconf import OmegaConf  
from gflownet.policy.base import Policy  
from gflownet.proxy.base import Proxy 
import torch
torch.distributions.Distribution.set_default_validate_args(False)

# --- PROXY CLASS ---
# Inherit from Proxy so it has .setup() and other required methods
class SimpleInitProxy(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self, states):
        # Return a constant energy value. Must be a numpy array of floats.
        return np.full(len(states), -50.0, dtype=np.float32)

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

    # 1. Instantiate the REAL MMFF Proxy
    real_mmff_proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    # 2. Instantiate the FAST DUMMY Proxy
    fast_init_proxy = SimpleInitProxy(
        device=config.device, 
        float_precision=config.float_precision
    )
    # 3. Instantiate Environment using the FAST Proxy (for buffer generation speed)
    env = hydra.utils.instantiate(
        config.env,
        proxy=fast_init_proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    # 4. Policy configs
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")

    # [Policy Instantiation Block - Your previously corrected code]
    # -------------------------------------------------------------------------
    # DIRECT INITIALIZATION (With Config Merging)
    # -------------------------------------------------------------------------
    
    fwd_params = OmegaConf.to_container(forward_config, resolve=True)
    bwd_params = OmegaConf.to_container(backward_config, resolve=True)

    def prepare_config(params):
        shared = params.pop('config', {}) or {}
        params.pop('_target_', None)
        flat_config = shared.copy()
        flat_config.update(params)
        return OmegaConf.create(flat_config), params

    fwd_config_obj, fwd_kwargs = prepare_config(fwd_params)
    bwd_config_obj, bwd_kwargs = prepare_config(bwd_params)

    forward_policy = Policy(
        config=fwd_config_obj,
        env=env, # Env uses FAST proxy here
        device=config.device,
        float_precision=config.float_precision,
        base=None,
        **fwd_kwargs 
    )

    backward_policy = Policy(
        config=bwd_config_obj,
        env=env, # Env uses FAST proxy here
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
        **bwd_kwargs
    )
    # -------------------------------------------------------------------------
    # 5. INITIALIZE GFLOWNET (With FAST PROXY still active!)
    # -------------------------------------------------------------------------
    
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env, # env.proxy is still 'fast_init_proxy' here!
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        buffer=config.env.buffer,
        logger=logger,
    )
    
    # -------------------------------------------------------------------------
    # 6. RESTORE THE REAL PROXY (After Buffer is filled)
    # ------------------------------------------------------------------------
    # update the proxy in the environment object held by the agent
    # (Since env is passed by reference, this updates it everywhere)
    env.proxy = real_mmff_proxy
    
    # Just to be safe, update the reference inside the agent explicitly
    gflownet.env.proxy = real_mmff_proxy

    # -------------------------------------------------------------------------
    # 7. TRAIN
    # -------------------------------------------------------------------------
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
