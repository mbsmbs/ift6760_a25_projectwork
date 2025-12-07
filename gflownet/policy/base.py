import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, VonMises, Categorical, MixtureSameFamily
from omegaconf import OmegaConf

from gflownet.utils.common import set_device, set_float_precision

# --- INSERT START: Helper Classes ---
class HeterogeneousPolicyHead(nn.Module):
    def __init__(
        self, 
        backbone_dim: int, 
        n_torsions: int, 
        n_lengths_angles: int, 
        n_components: int = 3
    ):
        super().__init__()
        self.n_torsions = n_torsions
        self.n_extras = n_lengths_angles
        self.n_components = n_components

        # --- HEAD 1: Torsions (Mixture of Von Mises) ---
        if n_torsions > 0:
            # 3 parameters per component: Weight, Concentration, Location
            self.torsion_out = nn.Linear(backbone_dim, n_torsions * n_components * 3)

        # --- HEAD 2: Lengths & Angles (Gaussian) ---
        if n_lengths_angles > 0:
            # 2 parameters: Mean (mu) and Log Scale (log_sigma)
            self.geometry_out = nn.Linear(backbone_dim, n_lengths_angles * 2)

    def forward(self, embedding):
        batch_size = embedding.shape[0]
        outputs = []

        # 1. Torsions
        if self.n_torsions > 0:
            # Output: [batch, n_torsions, n_components, 3]
            raw_t = self.torsion_out(embedding).view(batch_size, self.n_torsions, self.n_components, 3)
            
            # Apply activations immediately
            logits = raw_t[..., 0]
            # Map locs to [-pi, pi]
            locs = torch.tanh(raw_t[..., 1]) * torch.pi 
            # Softplus for concentration
            # concs = F.softplus(raw_t[..., 2]) + 0.1 
            #print(f"concs")
            # Force concentration to be at least 1.0 to avoid slow rejection sampling
            concs = F.softplus(raw_t[..., 2]) + 1.0
            
            # Flatten back to [batch, n_torsions * n_comp * 3]
            # We stack them so we can slice easily later: [logits, locs, concs]
            t_out = torch.stack([logits, locs, concs], dim=-1).reshape(batch_size, -1)
            outputs.append(t_out)

        # 2. Geometry
        if self.n_extras > 0:
            # Output: [batch, n_extras, 2]
            raw_g = self.geometry_out(embedding).view(batch_size, self.n_extras, 2)
            
            # Activations
            mu = torch.tanh(raw_g[..., 0]) # Delta form (-1 to 1)
            sigma = torch.sigmoid(raw_g[..., 1]) * 0.5 + 1e-4 # Small variance
            
            # Flatten to [batch, n_extras * 2]
            g_out = torch.stack([mu, sigma], dim=-1).reshape(batch_size, -1)
            outputs.append(g_out)

        # Concatenate everything into one big vector
        # This satisfies gflownet.py which expects a single tensor
        #print("finished forward pass")
        return torch.cat(outputs, dim=-1)

#--- INSERT END ---

class Policy:
    def __init__(self, config=None, env=None, device=None, float_precision=32, base=None, **kwargs):
        # -----------------------------------------------------------
        # DEBUGGING BLOCK: See what Hydra is actually passing
        # -----------------------------------------------------------
        if env is None:
            # Try to find env in kwargs if it wasn't passed positionally
            env = kwargs.get('env')

        if env is None:
            print("\n!!! DEBUG: Policy __init__ failed to find 'env' !!!")
            print(f"Direct 'env' arg is: {env}")
            print(f"Available kwargs keys: {list(kwargs.keys())}")
            # print(f"Config object: {config}")
            raise ValueError("Policy initialized without an 'env' argument! Check main.py instantiation.")
            
        # -----------------------------------------------------------
        # CONFIG SETUP
        # -----------------------------------------------------------
        # 1. Fallback: If config is missing, try to create it from kwargs
        if config is None:
             if kwargs:
                 config = OmegaConf.create(kwargs)
             else:
                 config = OmegaConf.create({})

        # 2. Standard Setup
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        
        # Dimensions from Env
        self.state_dim = env.policy_input_dim
        self.fixed_output = torch.tensor(env.fixed_policy_output).to(
            dtype=self.float, device=self.device
        )
        self.random_output = torch.tensor(env.random_policy_output).to(
            dtype=self.float, device=self.device
        )
        self.output_dim = len(self.fixed_output)
        self.base = base
        
        # Geometric Dimensions (Safe getattr)
        self.n_torsions = getattr(env, "n_torsion_angles", 0)
        self.n_bond_lengths = getattr(env, "n_bond_lengths", 0)
        self.n_bond_angles = getattr(env, "n_bond_angles", 0)

        self.parse_config(config)
        self.instantiate()


    def parse_config(self, config):
        # If config is null, default to uniform
        if config is None:
            config = OmegaConf.create()
            config.type = "uniform"
        if "checkpoint" in config:
            self.checkpoint = config.checkpoint
        else:
            self.checkpoint = None
        if "shared_weights" in config:
            self.shared_weights = config.shared_weights
        else:
            self.shared_weights = False
        if "n_hid" in config:
            self.n_hid = config.n_hid
        else:
            self.n_hid = None
        if "n_layers" in config:
            self.n_layers = config.n_layers
        else:
            self.n_layers = None
        if "tail" in config:
            self.tail = config.tail
        else:
            self.tail = []
        if "type" in config:
            self.type = config.type
        elif self.shared_weights:
            self.type = self.base.type
        else:
            raise ValueError("Policy type must be defined if shared_weights is False")

    def instantiate(self):
        if self.type == "fixed":
            self.model = self.fixed_distribution1
            self.is_model = False
        elif self.type == "uniform":
            self.model = self.uniform_distribution
            self.is_model = False
        elif self.type == "mlp":
            self.model = self.make_mlp(nn.LeakyReLU()).to(self.device)
            self.is_model = True
        
        # <--- CHANGED: Pass n_components explicitly ---
        elif self.type == "heterogeneous":
            # 1. Build Backbone
            backbone = self.make_backbone(nn.LeakyReLU())
            
            # 2. FORCE N_COMPONENTS TO 5
            # We know from config/env/conformer.yaml that n_comp is 5.
            n_components = 5
            print(f"[DEBUG POLICY] Forcing n_components = {n_components} to match Environment.")

            # 3. Build Head
            head = HeterogeneousPolicyHead(
                backbone_dim=self.n_hid,
                n_torsions=self.n_torsions,
                n_lengths_angles=self.n_bond_lengths + self.n_bond_angles,
                n_components=n_components 
            )
            
            # 4. Combine
            self.model = nn.Sequential(backbone, head).to(self.device)
            self.is_model = True
        # -----------------------------------------------------
        else:
            raise ValueError(f"Policy model type {self.type} not defined")

    def __call__(self, states):
        return self.model(states)

    # <--- CHANGED: Helper to build just the hidden layers ---
    def make_backbone(self, activation):
        """
        Creates the MLP layers up to the last hidden layer.
        Does NOT include the output layer.
        """
        layers_dim = [self.state_dim] + [self.n_hid] * self.n_layers
        
        layers = []
        for n, (idim, odim) in enumerate(zip(layers_dim, layers_dim[1:])):
            layers.append(nn.Linear(idim, odim))
            layers.append(activation)
            
        return nn.Sequential(*layers)
    # -------------------------------------------------------

    def make_mlp(self, activation):
        """
        Defines an MLP with no top layer activation
        If share_weight == True,
            baseModel (the model with which weights are to be shared) must be provided
        Args
        ----
        layers_dim : list
            Dimensionality of each layer
        activation : Activation
            Activation function
        """
        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp
        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim] + [self.n_hid] * self.n_layers + [(self.output_dim)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                    + self.tail
                )
            )
            return mlp
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )

    def fixed_distribution(self, states):
        """
        Returns the fixed distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.fixed_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def random_distribution(self, states):
        """
        Returns the random distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.random_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def uniform_distribution(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution
        Args: states: tensor
        """
        return torch.ones(
            (len(states), self.output_dim), dtype=self.float, device=self.device
        )
