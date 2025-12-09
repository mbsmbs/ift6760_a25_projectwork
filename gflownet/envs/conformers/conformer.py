import copy
from typing import List, Optional, Tuple, Union

import dgl
import numpy as np
import numpy.typing as npt
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.distributions import Uniform, Normal, VonMises, Categorical, MixtureSameFamily
import torch.nn.functional as F

# Dummy replacement for torchtyping to avoid dependency issues
class DummyTensorType:
    def __class_getitem__(cls, item):
        return object
TensorType = DummyTensorType

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.molecule.constants import ad_atom_types
from gflownet.utils.molecule.featurizer import MolDGLFeaturizer
from gflownet.utils.molecule.rdkit_conformer import RDKitConformer
from gflownet.utils.molecule.rotatable_bonds import find_rotor_from_smiles

# Predefined SMILES list (kept from your original code)
PREDEFINED_SMILES = [
    "O=C(c1ccccc1)c1ccc2c(c1)OCCOCCOCCOCCO2",
    "O=S(=O)(NN=C1CCCCCC1)c1ccc(Cl)cc1",
    "O=C(NC1CCCCC1)N1CCN(C2CCCCC2)CC1",
    "O=C(COc1ccc(Cl)cc1[N+](=O)[O-])N1CCCCCC1",
    "O=C(Nc1ccc(N2CCN(C(=O)c3ccccc3)CC2)cc1)c1cccs1",
    # ... (Truncated for brevity, the full list is assumed to be here)
    "O=C(NCCCN1CCN(CCCNC(=O)c2ccc3c(c2)OCO3)CC1)c1ccc2c(c1)OCO2",
]

class MoleculeHeterogeneousDistribution:
    def __init__(self, torsion_dist, geometry_dist):
        self.torsion_dist = torsion_dist
        self.geometry_dist = geometry_dist

    def sample(self, sample_shape=torch.Size()):
        samples = []
        if self.torsion_dist is not None:
            samples.append(self.torsion_dist.sample(sample_shape))
        if self.geometry_dist is not None:
            samples.append(self.geometry_dist.sample(sample_shape))
        return torch.cat(samples, dim=-1)

    def log_prob(self, value):
        log_prob_sum = 0
        idx = 0
        if self.torsion_dist is not None:
            n_t = self.torsion_dist.batch_shape[-1]
            t_val = value[..., :n_t]
            log_prob_sum = log_prob_sum + self.torsion_dist.log_prob(t_val).sum(-1)
            idx += n_t
        if self.geometry_dist is not None:
            g_val = value[..., idx:]
            log_prob_sum = log_prob_sum + self.geometry_dist.log_prob(g_val).sum(-1)
        return log_prob_sum

class Conformer(ContinuousTorus):
    """
    Extension of continuous torus to conformer generation.
    """

    def __init__(
        self,
        smiles: Union[str, int],
        n_torsion_angles: Optional[int] = 2,
        torsion_indices: Optional[List[int]] = None,
        policy_type: str = "mlp",
        remove_hs: bool = True,
        flex_bond_lengths: Optional[List[Tuple[int, int]]] = None,
        flex_bond_angles: Optional[List[Tuple[int, int, int]]] = None,
        length_scale: float = 0.05,
        angle_scale: float = 0.3,
        **kwargs,
    ):
        # 1. Extract config for distributions
        self.n_comp = kwargs.get("n_comp", 3)
        self.fixed_distribution = kwargs.get("fixed_distribution", {
            "vonmises_mean": 0.0, "vonmises_concentration": 0.5
        })
        self.random_distribution = kwargs.get("random_distribution", {
            "vonmises_mean": 0.0, "vonmises_concentration": 0.001
        })

        # Handle predefined smiles
        if isinstance(smiles, int) and smiles < len(PREDEFINED_SMILES):
            smiles = PREDEFINED_SMILES[smiles]
        elif isinstance(smiles, int):
            # Fallback if index is out of bounds
            smiles = PREDEFINED_SMILES[0]

        if torsion_indices is None:
            # Simple heuristic for specific tests, generic fallback otherwise
            if smiles == "CC(C(=O)NC)NC(=O)C" and n_torsion_angles == 2:
                torsion_indices = [0, 1]
            elif n_torsion_angles == -1:
                torsion_indices = None
            else:
                torsion_indices = list(range(n_torsion_angles)) if n_torsion_angles else []

        self.smiles = smiles
        self.torsion_indices = torsion_indices

        # 2. Setup Basic Geometry
        self.atom_positions = Conformer._get_positions(self.smiles)
        self.torsion_angles = Conformer._get_torsion_angles(
            self.smiles, self.torsion_indices
        )
        
        # 3. Setup Flexible Bonds & Angles
        tmp_mol = Chem.MolFromSmiles(self.smiles)
        tmp_mol = Chem.AddHs(tmp_mol)

        if flex_bond_lengths is None:
            self.flex_bond_lengths = Conformer.find_flexible_bonds(tmp_mol) 
        else:
            self.flex_bond_lengths = flex_bond_lengths

        if flex_bond_angles is None:
            self.flex_bond_angles = Conformer.find_flexible_angles(tmp_mol, self.flex_bond_lengths)
        else:
            self.flex_bond_angles = flex_bond_angles

        self.length_scale = length_scale
        self.angle_scale = angle_scale
        self.n_bond_lengths = len(self.flex_bond_lengths)
        self.n_bond_angles = len(self.flex_bond_angles)
        
        # 4. Create Conformer Object
        # call set_conformer() only once, after all definitions are ready.
        self.set_conformer() 

        # 5. Store Reference Values
        if self.n_bond_lengths > 0:
            self.ref_bond_lengths = self.conformer.get_bond_length_vector(self.flex_bond_lengths)
        else:
            self.ref_bond_lengths = np.zeros(0, dtype=float)

        if self.n_bond_angles > 0:
            self.ref_bond_angles = self.conformer.get_angle_vector(self.flex_bond_angles)
        else:
            self.ref_bond_angles = np.zeros(0, dtype=float)

        self.n_torsion_angles = len(self.torsion_angles)
        internal_dim = self.n_torsion_angles + self.n_bond_lengths + self.n_bond_angles

        # 6. Calculate Policy Outputs
        self.fixed_policy_output = self.get_policy_output(self.fixed_distribution)
        self.random_policy_output = self.get_policy_output(self.random_distribution)

        # Convert to Tensors immediately
        dev = getattr(self, 'device', 'cpu') 
        self.fixed_policy_output_list = self.get_policy_output(self.fixed_distribution) # Store as list
        self.random_policy_output_list = self.get_policy_output(self.random_distribution) # Store as list

        # 7. Parent Init (Passes the length 278 list/tensor to the parent)
        super().__init__(
            n_dim=internal_dim, 
            fixed_policy_output=self.fixed_policy_output, 
            random_policy_output=self.random_policy_output, 
            **kwargs
        )

        # 7.5. FINAL SYNC: Overwrite the parent's potentially incorrect fixed_output tensor
        # This ensures the Policy class (which uses env.fixed_output) gets the correct size.
        dev = getattr(self, 'device', 'cpu') 
        self.fixed_output = torch.tensor(
            self.fixed_policy_output_list, 
            dtype=torch.float32, 
            device=dev
        )
        self.random_output = torch.tensor(
            self.random_policy_output_list, 
            dtype=torch.float32, 
            device=dev
        )
        # ensure the environment's internal dimension is correct:
        # Assuming the parent class sets self.output_dim. force it to the correct length.
        self.output_dim = len(self.fixed_output)

        self.max_traj_length = kwargs.get("length_traj", 10)

        # 8. Remaining Setup
        self.statebatch2oracle = self.statebatch2proxy
        self.statetorch2oracle = self.statetorch2proxy
        if policy_type == "gnn":
            self.statebatch2policy = self.statebatch2policy_gnn
        elif policy_type != "mlp":
            raise ValueError(f"Unrecognized policy_type = {policy_type}")

        self.graph = MolDGLFeaturizer(ad_atom_types).mol2dgl(self.conformer.rdk_mol)
        rotatable_edges = [ta[1:3] for ta in self.torsion_angles]
        for i in range(self.graph.num_edges()):
            if (self.graph.edges()[0][i].item(), self.graph.edges()[1][i].item()) not in rotatable_edges:
                self.graph.edata["rotatable_edges"][i] = False

        self.remove_hs = remove_hs
        self.hs = torch.where(self.graph.ndata["atom_features"][:, 0] == 1)[0]
        self.non_hs = torch.where(self.graph.ndata["atom_features"][:, 0] != 1)[0]
        if remove_hs:
            self.graph = dgl.remove_nodes(self.graph, self.hs)

        self.sync_conformer_with_state()

    def set_conformer(self, state: Optional[List] = None) -> RDKitConformer:
        self.conformer = RDKitConformer(
            self.atom_positions,
            self.smiles,
            self.torsion_angles,
            freely_variable_bonds=self.flex_bond_lengths,
            freely_variable_angles=self.flex_bond_angles
        )
        if state is not None:
            self.sync_conformer_with_state(state)
        return self.conformer

    def get_log_jacobian(self, state):
        self.sync_conformer_with_state(state)
        lengths = self.conformer.get_bond_length_vector(self.flex_bond_lengths)
        angles = self.conformer.get_angle_vector(self.flex_bond_angles)
        log_r_term = 2 * np.sum(np.log(lengths))
        log_theta_term = np.sum(np.log(np.sin(angles)))
        return log_r_term + log_theta_term

    # In gflownet/envs/conformers/conformer.py, modify the _get_positions method:

    # In gflownet/envs/conformers/conformer.py

    @staticmethod
    def _get_positions(smiles: str) -> npt.NDArray:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # 1. Embed the molecule (initial guess)
        res = AllChem.EmbedMolecule(mol, randomSeed=0)
        
        # --- CRITICAL FIX: MINIMIZE AND CHECK FOR SUCCESS ---
        if res == 0:
            try:
                # The optimization modifies 'mol' in place.
                # We use maxIters=500 for robustness.
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500) 
                print("INFO: Initial RDKit geometry successfully minimized and positions updated.")
            except Exception as e:
                print(f"WARNING: MMFF Optimization failed during initialization: {e}")
                pass
        # ---------------------------------------------
        
        # 2. Extract and return the optimized positions from the modified 'mol' object.
        # This ensures we are getting the low-energy coordinates.
        # If optimization failed, this returns the strained embedded coordinates.
        return mol.GetConformer().GetPositions()

    @staticmethod
    def _get_torsion_angles(smiles: str, indices: Optional[List[int]]) -> List[Tuple[int]]:
        torsion_angles = find_rotor_from_smiles(smiles)
        if indices is not None and len(indices) > 0:
            # Filter indices that are out of bounds
            valid_indices = [i for i in indices if i < len(torsion_angles)]
            torsion_angles = [torsion_angles[i] for i in valid_indices]
        return torsion_angles

    def sync_conformer_with_state(self, state: List = None):
        if state is None:
            state = self.state

        state = np.asarray(state, dtype=float)
        
        # SAFETY: If state contains NaNs, replace them to prevent RDKit crash
        if np.isnan(state).any():
            state = np.nan_to_num(state, nan=0.0)

        internal_dim = (self.n_torsion_angles + self.n_bond_lengths + self.n_bond_angles)

        if state.shape[0] == internal_dim + 1:
            internal = state[:-1]
        elif state.shape[0] == internal_dim:
            internal = state
        else:
            # Fallback for empty/wrong states
            return self.conformer

        k0 = self.n_torsion_angles
        k1 = k0 + self.n_bond_lengths
        k2 = k1 + self.n_bond_angles

        z_tors = internal[:k0]
        z_len = internal[k0:k1] if self.n_bond_lengths > 0 else None
        z_ang = internal[k1:k2] if self.n_bond_angles > 0 else None

        if self.n_torsion_angles > 0:
            z_tors_deg = np.degrees(z_tors)
            self.conformer.set_torsion_vector(self.torsion_angles, z_tors_deg)

        if self.n_bond_lengths > 0 and z_len is not None:
            new_lengths = self.ref_bond_lengths + self.length_scale * z_len
            self.conformer.set_bond_length_vector(self.flex_bond_lengths, new_lengths)

        if self.n_bond_angles > 0 and z_ang is not None:
            new_angles = self.ref_bond_angles + self.angle_scale * z_ang
            self.conformer.set_angle_vector(self.flex_bond_angles, new_angles)

        return self.conformer

    def statebatch2proxy(self, states: List[List]) -> npt.NDArray:
        states_proxy = []
        for st in states:
            conf = self.sync_conformer_with_state(st)
            states_proxy.append(
                np.concatenate(
                    [
                        conf.get_atomic_numbers()[..., np.newaxis],
                        conf.get_atom_positions(),
                    ],
                    axis=1,
                )
            )
        return np.array(states_proxy)

    def statetorch2proxy(self, states: TensorType["batch", "state_dim"]) -> npt.NDArray:
        return self.statebatch2proxy(states.cpu().numpy())

    def statebatch2policy_gnn(self, states: List[List]) -> npt.NDArray[np.float32]:
        policy_input = []
        for state in states:
            conformer = self.sync_conformer_with_state(state)
            positions = conformer.get_atom_positions()
            if self.remove_hs:
                positions = positions[self.non_hs]
            # Ensure time dimension is present; if not, append 0
            t = state[-1] if len(state) > (self.n_dim) else 0.0
            policy_input.append(
                np.concatenate(
                    [positions, np.full((positions.shape[0], 1), t)],
                    axis=1,
                )
            )
        return np.array(policy_input)

    def statebatch2kde(self, states: List[List]) -> npt.NDArray[np.float32]:
        return np.array(states)[:, :-1]

    def statetorch2kde(self, states: TensorType["batch_size", "state_dim"]) -> TensorType["batch_size", "state_proxy_dim"]:
        return states.cpu().numpy()[:, :-1]

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_instance = cls.__new__(cls)
        for attr_name, attr_value in self.__dict__.items():
            if attr_name != "conformer":
                setattr(new_instance, attr_name, copy.copy(attr_value))
        new_instance.conformer = self.conformer
        return new_instance

    # --- REWARD METHOD ---
    def reward_batch(self, states: List[List]) -> TensorType["batch"]:
        """
        Computes the geometric reward: exp( -Energy + log_Jacobian )
        Robust version: Handles NaNs, Infs, and Overflow/Underflow.
        """
        # 1. Convert states to RDKit Mols/Conformers & Get Energies
        energies = self.proxy(states) 
        
        # 2. Compute Jacobians
        log_jacobians = []
        for state in states:
            try:
                log_jacobians.append(self.get_log_jacobian(state))
            except:
                log_jacobians.append(0.0) # Fallback
        
        # Convert to tensors
        log_jacobians = torch.tensor(log_jacobians, device=self.device, dtype=torch.float)
        if isinstance(energies, np.ndarray):
            energies = torch.tensor(energies, device=self.device, dtype=torch.float)

        # --- SAFETY BLOCK START ---
        
        # A. Sanitize Energies (Crucial for Physics)
        # Replace NaNs with a "bad energy" value (e.g., 1000.0)
        energies = torch.nan_to_num(energies, nan=1000.0, posinf=1000.0, neginf=-1000.0)
        energies = torch.clamp(energies, min=-1000.0, max=1000.0)

        # B. Sanitize Jacobians (Crucial for Coordinate Singularities)
        log_jacobians = torch.nan_to_num(log_jacobians, nan=0.0, posinf=100.0, neginf=-100.0)
        log_jacobians = torch.clamp(log_jacobians, min=-100.0, max=100.0)
        
        # --- SAFETY BLOCK END ---

        # 3. Combine: log_R = -E/T + log_J
        beta = self.reward_beta 
        log_reward = -energies * beta + log_jacobians
        
        # C. Prevent Exponential Explosion
        # exp(88) is approx float32 max. Clamp safely below that.
        log_reward = torch.clamp(log_reward, max=80.0)
        
        # 4. Exponentiate
        reward = torch.exp(log_reward)
        
        # Final Safety: Clip very small rewards
        return torch.clamp(reward, min=self.min_reward)

    def reward2proxy(self, reward: TensorType["batch"]) -> TensorType["batch"]:
        return -torch.log(reward + 1e-10) / self.reward_beta

    def parse_policy_output(self, policy_outputs: TensorType["batch", "dim"]):
        batch_size = policy_outputs.shape[0]
        idx = 0
        
        # 1. Reconstruct Torsions
        torsion_dist = None
        if self.n_torsion_angles > 0:
            n_t = self.n_torsion_angles
            n_c = self.n_comp 
            block_size = n_t * n_c * 3
            t_params = policy_outputs[:, idx : idx + block_size]
            idx += block_size
            
            t_params = t_params.view(batch_size, n_t, n_c, 3)
            logits = t_params[..., 0]
            locs = t_params[..., 1]
            concs = t_params[..., 2]
            
            # Ensure concentration is positive
            concs = F.softplus(concs) + 0.001

            mix = Categorical(logits=logits)
            comp = VonMises(locs, concs)
            torsion_dist = MixtureSameFamily(mix, comp)

        # 2. Reconstruct Geometry
        geometry_dist = None
        n_geo = self.n_bond_lengths + self.n_bond_angles
        if n_geo > 0:
            block_size = n_geo * 2
            g_params = policy_outputs[:, idx : idx + block_size]
            idx += block_size
            g_params = g_params.view(batch_size, n_geo, 2)
            
            mu = g_params[..., 0]
            sigma = g_params[..., 1]
            sigma = F.softplus(sigma) + 0.001
            
            geometry_dist = Normal(mu, sigma)

        return MoleculeHeterogeneousDistribution(torsion_dist, geometry_dist)

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["batch", "policy_output_dim"],
        mask: Optional[TensorType["batch", "action_space_dim"]] = None,
        states: Optional[List] = None,
        is_backward: bool = False,
        sampling_method: str = "policy",
        temperature: float = 1.0,
    ) -> Tuple[List[Tuple], TensorType["batch"]]:
        
        if not torch.is_tensor(policy_outputs):
            policy_outputs = torch.tensor(policy_outputs, device=self.device, dtype=torch.float)

        # Detect Fast Path / Random
        use_fast_path = False
        if sampling_method == "random":
            use_fast_path = True
        elif self.n_torsion_angles > 0:
            # Check concentration. If < 0.1, it's the initialization policy.
            # We use a very low threshold to be safe.
            first_concentration = policy_outputs[0, 2]
            if first_concentration < 0.1: 
                use_fast_path = True

        if use_fast_path:
            # FAST PATH: Uniform Sampling to break identical sample loop
            parts = []
            if self.n_torsion_angles > 0:
                 # Sample uniformly [0, 2pi]
                uniform_tors = Uniform(
                    torch.zeros(self.n_torsion_angles, device=self.device), 
                    2 * np.pi * torch.ones(self.n_torsion_angles, device=self.device)
                )
                parts.append(uniform_tors.sample(sample_shape=[policy_outputs.shape[0]]))
            
            n_geo = self.n_bond_lengths + self.n_bond_angles
            if n_geo > 0:
                uniform_geo = Uniform(
                    -torch.ones(n_geo, device=self.device), 
                    torch.ones(n_geo, device=self.device)
                )
                parts.append(uniform_geo.sample(sample_shape=[policy_outputs.shape[0]]))
            
            actions_tensor = torch.cat(parts, dim=-1)
            
            # Approximate logprobs for uniform (constant)
            logprobs = torch.zeros(policy_outputs.shape[0], device=self.device)
            
        else:
            dist = self.parse_policy_output(policy_outputs)
            actions_tensor = dist.sample()
            logprobs = dist.log_prob(actions_tensor)
        
        actions = [tuple(x.tolist()) for x in actions_tensor]
        return actions, logprobs

    def get_logprobs(self, policy_outputs, actions, mask, states=None, is_backward=False):
        dist = self.parse_policy_output(policy_outputs)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float)
        return dist.log_prob(actions)

    def get_policy_output(self, params: dict) -> List[float]:
        n_t = self.n_torsion_angles
        n_c = self.n_comp
        torsion_out = []
        
        if n_t > 0:
            # Flattened params for Mixture of Von Mises
            single_angle_params = []
            for _ in range(n_c):
                # [logit, loc, conc]
                single_angle_params.extend([0.0, params['vonmises_mean'], params['vonmises_concentration']])
            torsion_out = single_angle_params * n_t

        n_geo = self.n_bond_lengths + self.n_bond_angles
        geometry_out = []
        if n_geo > 0:
             # [mu, sigma]
            single_geo_params = [0.0, 0.01]
            geometry_out = single_geo_params * n_geo

        return torsion_out + geometry_out

    def _get_next_state(self, state, action):
        s = torch.tensor(state, device=self.device, dtype=torch.float)
        a = torch.tensor(action, device=self.device, dtype=torch.float)
        n_dims = len(a)
        
        physical_state = s[:n_dims] + a
        
        if self.n_torsion_angles > 0:
            physical_state[:self.n_torsion_angles] = torch.remainder(
                physical_state[:self.n_torsion_angles], 2 * np.pi
            )

        if n_dims > self.n_torsion_angles:
            physical_state[self.n_torsion_angles:] = torch.clamp(
                physical_state[self.n_torsion_angles:], -5.0, 5.0
            )
            
        if len(s) > n_dims:
            final_state = torch.cat([physical_state, s[n_dims:]])
        else:
            final_state = physical_state
            
        return final_state.tolist()

    # --- RESET / STEP (The Critical Fixes) ---

    def reset(self, env_id: Union[int, str] = None):
        super().reset(env_id)
        
        
        if hasattr(self.source, 'tolist'):
            base_state = self.source.tolist()
        else:
            base_state = list(self.source)
            
        internal_dim = self.n_torsion_angles + self.n_bond_lengths + self.n_bond_angles
        
        # Ensure correct state dimension including Time
        if len(base_state) == internal_dim:
            self.state = base_state + [0.0]
        elif len(base_state) == internal_dim + 1:
            self.state = base_state
            self.state[-1] = 0.0 
        else:
            # Fallback for unexpected shapes
            self.state = base_state[:internal_dim] + [0.0]

        self.n_actions = 0
        self.done = False
        self.sync_conformer_with_state(self.state)
        
        return self

    def step(self, action: Tuple[float], skip_mask_check: bool = False) -> Tuple[List[float], Tuple[float], bool]:
        self.state = self._get_next_state(self.state, action)
        self.n_actions += 1
        
        if self.n_actions >= self.max_traj_length:
            self.done = True
            
        return self.state, action, True

    def step_backwards(self, action, skip_mask_check=False):
        if self.done:
            self.done = False
            return self.state, self.eos, True
        return super().step_backwards(action, skip_mask_check)

    @staticmethod
    def find_flexible_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
        bonds = []
        for b in mol.GetBonds():
            if b.IsInRing(): continue
            bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        return bonds

    @staticmethod
    def find_flexible_angles(mol: Chem.Mol, flexible_bonds: List[Tuple[int, int]] = None) -> List[Tuple[int, int, int]]:
        if flexible_bonds is None:
            flexible_bonds = Conformer.find_flexible_bonds(mol)
        flex_bond_set = set()
        for b in flexible_bonds:
            flex_bond_set.add(tuple(sorted(b)))

        angles = []
        for atom in mol.GetAtoms():
            j = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) < 2: continue
            
            import itertools
            for n1, n2 in itertools.combinations(neighbors, 2):
                bond1 = tuple(sorted((n1, j)))
                bond2 = tuple(sorted((n2, j)))
                if bond1 in flex_bond_set or bond2 in flex_bond_set:
                    angles.append((n1, j, n2))
        return angles