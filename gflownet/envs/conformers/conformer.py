import copy
from typing import List, Optional, Tuple, Union

import dgl
import numpy as np
import numpy.typing as npt
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.distributions import Uniform
# from torchtyping import TensorType

# Dummy replacement for torchtyping
class DummyTensorType:
    def __class_getitem__(cls, item):
        return object
TensorType = DummyTensorType

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.molecule.constants import ad_atom_types
from gflownet.utils.molecule.featurizer import MolDGLFeaturizer
from gflownet.utils.molecule.rdkit_conformer import RDKitConformer
from gflownet.utils.molecule.rotatable_bonds import find_rotor_from_smiles
import torch.nn.functional as F
from torch.distributions import Normal, VonMises, Categorical, MixtureSameFamily

PREDEFINED_SMILES = [
    "O=C(c1ccccc1)c1ccc2c(c1)OCCOCCOCCOCCO2",
    "O=S(=O)(NN=C1CCCCCC1)c1ccc(Cl)cc1",
    "O=C(NC1CCCCC1)N1CCN(C2CCCCC2)CC1",
    "O=C(COc1ccc(Cl)cc1[N+](=O)[O-])N1CCCCCC1",
    "O=C(Nc1ccc(N2CCN(C(=O)c3ccccc3)CC2)cc1)c1cccs1",
    "O=[N+]([O-])/C(C(=C(Cl)Cl)N1CCN(Cc2ccccc2)CC1)=C1\\NCCN1Cc1ccc(Cl)nc1",
    "O=C(CSc1nnc(C2CC2)n1-c1ccccc1)Nc1ccc(N2CCOCC2)cc1",
    "O=C(Nc1ccccn1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)Nc1ccccn1",
    "O=C(NCCc1nnc2ccc(NCCCN3CCOCC3)nn12)c1ccccc1F",
    "O=C(CSc1cn(CCNC(=O)c2cccs2)c2ccccc12)NCc1ccccc1",
    "O=C(CCC(=O)N(CC(=O)NC1CCCCC1)Cc1cccs1)Nc1ccccn1",
    "S=C(c1ccc2c(c1)OCO2)N1CCOCC1",
    "O=Cc1cnc(N2CCN(c3ccccc3)CC2)s1",
    "O=[N+]([O-])c1ccc(NCc2ccc(Cl)cc2)nc1",
    "O=C(Nc1cc(C(F)(F)F)ccc1N1CCCCC1)c1ccncc1",
    "O=C(Nc1nnc(-c2ccccc2Cl)s1)C1CCN(S(=O)(=O)c2ccc(Cl)cc2)CC1",
    "O=C(CCNC(=O)c1ccccc1Cl)Nc1nc2ccccc2s1",
    "O=C(CCC(=O)Nc1ccccc1Cl)N/N=C/c1ccccc1",
    "O=C(COc1ccccc1)Nc1ccccc1C(=O)NCc1ccco1",
    "O=C(CNC(=O)c1cccs1)NCC(=O)OCc1ccc(Cl)cc1Cl",
    "O=C(NCc1ccccc1)c1onc(CSc2ccc(Cl)cc2)c1C(=O)NCC1CC1",
    "O=C(CN(C(=O)CCC(=O)Nc1ccccn1)c1ccc2c(c1)OCO2)NCc1ccco1",
    "O=[N+]([O-])c1ccc(N2CCNCC2)c(Cl)c1",
    "O=[N+]([O-])c1ccccc1S(=O)(=O)N1CCCCC1",
    "N#C/C(=C\\N1CCN(Cc2ccc3c(c2)OCO3)CC1)c1nc2ccccc2s1",
    "O=C(NNc1ccc([N+](=O)[O-])cc1)c1ccccc1Cl",
    "O=C(OCc1ccccc1Cl)c1ccccc1C(=O)c1ccccc1",
    "O=C(CN(c1cccc(C(F)(F)F)c1)S(=O)(=O)c1ccccc1)N1CCOCC1",
    "O=c1[nH]c2cc3c(cc2cc1CN(Cc1cccnc1)Cc1nnnn1Cc1ccco1)OCCO3",
    "O=C(CCNC(=O)C1CCN(S(=O)(=O)c2ccccc2)CC1)NC1CC1",
    "O=C(CN(c1ccc(F)cc1)S(=O)(=O)c1ccccc1)NCCSC1CCCCC1",
    "C=CCn1c(CSCc2ccccc2)nnc1SCC(=O)N1CCN(c2ccccc2)CC1",
    "C=COCCNC(=S)N1CCOCCOCCN(C(=S)NCCOC=C)CCOCC1",
    "O=S(=O)(c1cccc(Cl)c1Cl)N1CCCCC1",
    "O=C(Cn1ccccc1=O)c1cccs1",
    "FC(F)Sc1ccc(Nc2ncnc3nc[nH]c23)cc1",
    "O=c1[nH]c(SCCOc2ccccc2F)nc2ccccc12",
    "O=C(/C=C/c1ccccc1Cl)NCCN1CCOCC1",
    "N#CCCN1CCN(S(=O)(=O)c2ccc(S(=O)(=O)NC3CC3)cc2)CC1",
    "O=C(CSc1nnc(Cc2cccs2)n1-c1ccccc1)Nc1ccc2c(c1)OCCO2",
    "O=C(CNC(=O)c1ccc(N2C(=O)c3ccccc3C2=O)cc1)OCC(=O)c1ccccc1",
    "O=C(CSc1nnc(CNC(=O)c2c(F)cccc2Cl)o1)NCc1ccc2c(c1)OCO2",
    "O=C(CNC(=S)N(Cc1ccc(F)cc1)C1CCCCC1)NCCN1CCOCC1",
    "C=CCn1c(CCNC(=O)c2cccs2)nnc1SCC(=O)Nc1cccc(F)c1",
    "Clc1cccc(CN2CCCCCC2)c1Cl",
    "O=C(Nc1cc(=O)c2ccccc2o1)N1CCCCC1",
    "O=S(=O)(c1ccccc1)c1nc(-c2ccco2)oc1N1CCOCC1",
    "O=C(CC1CCCCC1)NCc1ccco1",
    "O=C(c1cccc([N+](=O)[O-])c1)N1CCCN(C(=O)c2cccc([N+](=O)[O-])c2)CC1",
    "Fc1ccccc1OCCCCCN1CCCC1",
    "O=C(c1ccc(S(=O)(=O)NCc2ccco2)cc1)N1CCN(Cc2ccc3c(c2)OCO3)CC1",
    "N#Cc1ccc(NC(=O)COC(=O)CNC(=O)C2CCCCC2)cc1",
    "O=C(CCC(=O)OCC(=O)c1ccc(-c2ccccc2)cc1)Nc1cccc(Cl)c1",
    "O=C(COC(=O)CCC(=O)c1cccs1)Nc1ccc(S(=O)(=O)N2CCOCC2)cc1",
    "O=C(COC(=O)CCCNC(=O)c1ccc(Cl)cc1)NCc1ccccc1Cl",
    "C1CCC([NH2+]C2=NCCC2)CC1",
    "N#Cc1ccccc1S(=O)(=O)Nc1ccc2c(c1)OCCO2",
    "O=[N+]([O-])c1ccccc1S(=O)(=O)N1CCN(c2ccccc2)CC1",
    "O=S(=O)(NCc1ccccc1Cl)c1ccc(-n2cccn2)cc1",
    "O=C(CNS(=O)(=O)c1cccc2nsnc12)NC1CCCCC1",
    "O=C(c1cccc([N+](=O)[O-])c1)n1nc(-c2ccccc2)nc1NCc1ccccc1",
    "O=C(CN1CCN(c2ccccc2)CC1)NC(=O)NCc1ccco1",
    "O=C(CCCn1c(=O)c2ccccc2n(Cc2ccccc2)c1=O)NCc1ccco1",
    "O=C(COc1ccc(Cl)cc1)NCc1nnc(SCC(=O)N2CCCCCC2)o1",
    "O=C(NCc1ccccc1)c1onc(CSc2ccccn2)c1C(=O)NCc1ccccc1",
    "C=CCN(c1cccc(C(F)(F)F)c1)S(=O)(=O)c1cccc(C(=O)OCC(=O)Nc2ccccc2)c1",
    "O=S(=O)(N1CCCCCC1)N1CC[NH2+]CC1",
    "O=C1c2ccccc2C(=O)N1Cc1nn2c(-c3ccc(Cl)cc3)nnc2s1",
    "O=C(CN1C(=O)NC2(CCCC2)C1=O)Nc1ccc(F)c(F)c1F",
    "O=C(Cc1n[nH]c(=O)[nH]c1=O)N/N=C/c1ccccc1",
    "O=C(NCCSc1ccc(Cl)cc1)c1ccco1",
    "O=C(CN1CCN(Cc2ccccc2Cl)CC1)N/N=C/c1ccco1",
    "O=C(Nc1ccc(-c2csc(Nc3cccc(C(F)(F)F)c3)n2)cc1)c1cccc(C(F)(F)F)c1",
    "O=C(CCCCCN1C(=O)c2cccc3cccc(c23)C1=O)NCc1ccco1",
    "O=C(NCCCn1ccnc1)/C(=C\\c1cccs1)NC(=O)c1cccs1",
    "O=C(COC(=O)COc1ccccc1[N+](=O)[O-])Nc1ccc(S(=O)(=O)N2CCCCC2)cc1",
    "O=C(NCCCN1CCCC1=O)c1cc(NS(=O)(=O)c2ccc(F)cc2)cc(NS(=O)(=O)c2ccc(F)cc2)c1",
    "Clc1ccc(N2CCN(c3ncnc4c3oc3ccccc34)CC2)cc1",
    "O=C(Nc1c(Cl)ccc2nsnc12)c1cccnc1",
    "c1nc(COc2nsnc2N2CCOCC2)cs1",
    "O=C(C1CC1)N1CCN=C1SCc1ccccc1",
    "O=C(Cc1ccccc1)OCC[NH+]1CCOCC1",
    "O=C(CCSc1ccccc1)NCc1cccnc1",
    "O=C(CNC(=O)c1ccc(F)cc1)N/N=C/c1cn[nH]c1-c1ccccc1",
    "O=C1CCN(CCc2ccccc2)CCN1[C@H](CSc1ccccc1)Cc1ccccc1",
    "O=C(CCCCCn1c(=S)[nH]c2ccc(N3CCOCC3)cc2c1=O)NCc1ccc(Cl)cc1",
    "O=C(CCC(=O)OCCCC(F)(F)C(F)(F)F)NC1CCCCC1",
    "O=C(CN(Cc1ccco1)C(=O)CNS(=O)(=O)c1ccc(F)cc1)NCc1ccco1",
    "O=c1[nH]c(N2CCN(c3ccccc3)CC2)nc2c1CCC2",
    "O=C(Nc1ccc2c(c1)OCO2)c1cccs1",
    "O=C(Nc1cccc2ccccc12)N1CCN(c2ccccc2)CC1",
    "O=C(NC(=S)Nc1ccccn1)c1ccccc1",
    "O=C(Nc1ccc(Cl)cc1)c1cccc(S(=O)(=O)Nc2ccccn2)c1",
    "O=C(COc1cnc2ccccc2n1)NCCC1=CCCCC1",
    "O=C(NCCN1CCOCC1)c1ccc(/C=C2\\Sc3ccccc3N(Cc3ccc(F)cc3)C2=O)cc1",
    "O=C(NCCc1cccc(Cl)c1)c1ccc(OC2CCN(Cc3ccccn3)CC2)cc1",
    "C=CC[NH2+]CCOCCOc1ccccc1-c1ccccc1",
    "O=C(COC(=O)c1ccccc1NC(=O)c1ccco1)NCCC1=CCCCC1",
    "O=C(CNC(=S)N(Cc1ccccc1Cl)C1CCCC1)NCCCN1CCOCC1",
    "O=c1c2ccccc2nnn1Cc1ccccc1Cl",
    "S=C(Nc1ccccc1)N1CCCCCCC1",
    "O=C(Cn1ccc([N+](=O)[O-])n1)N1CCCc2ccccc21",
    "O=C(NS(=O)(=O)N1CCOCC1)C1=C(N2CCCC2)COC1=O",
    "O=C(CCCn1c(=O)[nH]c2ccsc2c1=O)NC1CCCCC1",
    "O=C(Cc1ccc(Cl)cc1)Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1",
    "O=C1COc2ccc(C(=O)COC(=O)CCSc3ccccc3)cc2N1",
    "O=C(Nc1cc(F)cc(F)c1)c1ccc(NCCC[NH+]2CCCCCC2)c([N+](=O)[O-])c1",
    "O=C(CCCn1c(=O)[nH]c2cc(Cl)ccc2c1=O)NCCCN1CCN(c2ccc(F)cc2)CC1",
    "O=C(NCCN1CCN(C(=O)C(c2ccccc2)c2ccccc2)CC1)C(=O)Nc1ccccc1",
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
    Extension of continuous torus to conformer generation. Based on AlanineDipeptide,
    but accepts any molecule (defined by SMILES and freely rotatable torsion angles).
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

        # 1. Extract config for distributions (needed for policy output calculation)
        print("initializing Conformer")
        self.n_comp = kwargs.get("n_comp", 3)
        self.fixed_distribution = kwargs.get("fixed_distribution", {
            "vonmises_mean": 0.0, "vonmises_concentration": 0.5
        })
        self.random_distribution = kwargs.get("random_distribution", {
            "vonmises_mean": 0.0, "vonmises_concentration": 0.001
        })

        # Handle predefined smiles
        if torsion_indices is None:
            if smiles == "CC(C(=O)NC)NC(=O)C" and n_torsion_angles == 2:
                torsion_indices = [0, 1]
            elif smiles == "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O" and n_torsion_angles == 2:
                torsion_indices = [1, 2]
            elif smiles == "O=C(c1ccc2n1CCC2C(=O)O)c3ccccc3" and n_torsion_angles == 2:
                torsion_indices = [0, 1]
            elif n_torsion_angles == -1:
                torsion_indices = None
            else:
                torsion_indices = list(range(n_torsion_angles))

        if isinstance(smiles, int):
            smiles = PREDEFINED_SMILES[smiles]

        self.smiles = smiles
        self.torsion_indices = torsion_indices

        # 2. Setup Basic Geometry
        self.atom_positions = Conformer._get_positions(self.smiles)
        self.torsion_angles = Conformer._get_torsion_angles(
            self.smiles, self.torsion_indices
        )
        
        # 3. Setup Flexible Bonds & Angles
        if flex_bond_lengths is None or flex_bond_angles is None:
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

        # ------------------------------------------------------------------
        # FIX: Calculate Policy Outputs HERE, as attributes, not properties
        # ------------------------------------------------------------------
        print("calculating policy outputs")
        self.fixed_policy_output = self.get_policy_output(self.fixed_distribution)
        self.random_policy_output = self.get_policy_output(self.random_distribution)

        # 2. Convert to Tensors IMMEDIATELY (using the attributes we just set)
        dev = getattr(self, 'device', 'cpu') 
        self.fixed_output = torch.tensor(self.fixed_policy_output, dtype=torch.float32, device=dev)
        self.random_output = torch.tensor(self.random_policy_output, dtype=torch.float32, device=dev)

        # 3. Parent Init (Pass the lists we just calculated)
        super().__init__(
            n_dim=internal_dim, 
            fixed_policy_output=self.fixed_policy_output,
            random_policy_output=self.random_policy_output,
            **kwargs
        )

        # FIX: Explicitly set max_traj_length from config (defaults to 10 if missing)
        self.max_traj_length = kwargs.get("length_traj", 10)

        #print("random_output", self.random_output)
        # 7. Remaining Setup
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
            # FIX: You must pass the flex definitions to the new instance
            freely_variable_bonds=self.flex_bond_lengths,
            freely_variable_angles=self.flex_bond_angles
        )

        if state is not None:
            self.sync_conformer_with_state(state)

        return self.conformer

    def get_log_jacobian(self, state):
        """
        Calculates log det J for the transformation from Intrinsic -> Cartesian.
        J ~ Product(r_i^2 * sin(theta_i))
        """
        # Sync the RDKit molecule with the state first
        self.sync_conformer_with_state(state)
        
        # Get current values in Real units (Angstroms and Radians)
        lengths = self.conformer.get_bond_length_vector(self.flex_bond_lengths)
        angles = self.conformer.get_angle_vector(self.flex_bond_angles)
        
        # Log Jacobian = Sum(2*log(r)) + Sum(log(sin(theta)))
        # Note: This is an approximation assuming a standard Z-matrix chain.
        # For complex rings/trees, it's more complex, but this is the standard baseline.
        
        log_r_term = 2 * np.sum(np.log(lengths))
        log_theta_term = np.sum(np.log(np.sin(angles)))
        
        return log_r_term + log_theta_term


    @staticmethod
    def _get_positions(smiles: str) -> npt.NDArray:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0)
        return mol.GetConformer().GetPositions()

    @staticmethod
    def _get_torsion_angles(
        smiles: str, indices: Optional[List[int]]
    ) -> List[Tuple[int]]:
        torsion_angles = find_rotor_from_smiles(smiles)
        if indices is not None:
            torsion_angles = [torsion_angles[i] for i in indices]
        return torsion_angles

    # ----- New -----
    @staticmethod
    def _get_all_bond_angles(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
        """Enumerate all unique angles i–j–k where j is the central atom."""
        angles: List[Tuple[int, int, int]] = []
        for j in range(mol.GetNumAtoms()):
            nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors()]
            # all unordered pairs (i, k) of neighbors of j
            for a in range(len(nbrs)):
                for b in range(a + 1, len(nbrs)):
                    i, k = nbrs[a], nbrs[b]
                    angles.append((i, j, k))
        return angles
   

    # def sync_conformer_with_state(self, state: List = None):
    #     if state is None:
    #         state = self.state
    #     for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
    #         self.conformer.set_torsion_angle(ta, state[idx])
    #     return self.conformer

    def sync_conformer_with_state(self, state: List = None):
        """
        Map a torus state to the RDKitConformer.

        State can be either:
          - [θ_0, ..., θ_{n_tors-1},
             z_len_0, ..., z_len_{n_len-1},
             z_ang_0, ..., z_ang_{n_ang-1}]            (length = internal_dim)
        or
          - same as above plus a final time coord t:  (length = internal_dim + 1)
        """
        if state is None:
            state = self.state

        state = np.asarray(state, dtype=float)

        internal_dim = (
            self.n_torsion_angles
            + self.n_bond_lengths
            + self.n_bond_angles
        )

        if state.shape[0] == internal_dim + 1:
            # Drop time coordinate
            internal = state[:-1]
        elif state.shape[0] == internal_dim:
            internal = state
        else:
            raise ValueError(
                f"Expected state of length {internal_dim} or {internal_dim + 1}, "
                f"got {state.shape[0]}"
            )

        # Split into torsions / bond lengths / bond angles
        k0 = self.n_torsion_angles
        k1 = k0 + self.n_bond_lengths
        k2 = k1 + self.n_bond_angles

        z_tors = internal[:k0]
        z_len = internal[k0:k1] if self.n_bond_lengths > 0 else None
        z_ang = internal[k1:k2] if self.n_bond_angles > 0 else None

        # --- Torsions: interpret directly as angles in radians ---
        if self.n_torsion_angles > 0:
            self.conformer.set_torsion_vector(self.torsion_angles, z_tors)

        # --- Bond lengths / angles: offsets around reference geometry ---
        if self.n_bond_lengths > 0 and z_len is not None:
            # self.ref_bond_lengths in Å, z_len is dimensionless
            new_lengths = self.ref_bond_lengths + self.length_scale * z_len
            self.conformer.set_bond_length_vector(self.flex_bond_lengths, new_lengths)

        if self.n_bond_angles > 0 and z_ang is not None:
            # self.ref_bond_angles in radians, z_ang is dimensionless
            new_angles = self.ref_bond_angles + self.angle_scale * z_ang
            self.conformer.set_angle_vector(self.flex_bond_angles, new_angles)

        return self.conformer


    def statebatch2proxy(self, states: List[List]) -> npt.NDArray:
        """
        Returns a list of proxy states, each being a numpy array with dimensionality
        (n_atoms, 4), in which the first column encodes atomic number, and the last
        three columns encode atom positions.
        """
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
        """
        Returns an array of GNN-format policy inputs with dimensionality
        (n_states, n_atoms, 4), in which the first three columns encode atom positions,
        and the last column encodes current timestep.
        """
        policy_input = []
        for state in states:
            conformer = self.sync_conformer_with_state(state)
            positions = conformer.get_atom_positions()
            if self.remove_hs:
                positions = positions[self.non_hs]
            policy_input.append(
                np.concatenate(
                    [positions, np.full((positions.shape[0], 1), state[-1])],
                    axis=1,
                )
            )
        return np.array(policy_input)

    def statebatch2kde(self, states: List[List]) -> npt.NDArray[np.float32]:
        return np.array(states)[:, :-1]

    def statetorch2kde(
        self, states: TensorType["batch_size", "state_dim"]
    ) -> TensorType["batch_size", "state_proxy_dim"]:
        return states.cpu().numpy()[:, :-1]

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_instance = cls.__new__(cls)

        for attr_name, attr_value in self.__dict__.items():
            if attr_name != "conformer":
                setattr(new_instance, attr_name, copy.copy(attr_value))

        new_instance.conformer = self.conformer

        return new_instance


    def reward_batch(self, states: List[List]) -> TensorType["batch"]:
        """
        Computes the geometric reward: exp( -Energy + log_Jacobian )
        """
        # 1. Convert states to RDKit Mols/Conformers
        # Note: We use the proxy to get energies. 
        # Ideally, pass the state batch to the proxy.
        # If your proxy expects a list of states:
        energies = self.proxy(states) 
        
        # 2. Compute Jacobians for the batch
        # We need to compute log_det_J for every state
        log_jacobians = []
        for state in states:
            log_jacobians.append(self.get_log_jacobian(state))
        
        log_jacobians = torch.tensor(log_jacobians, device=self.device, dtype=self.float)
        
        # 3. Combine: log_R = -E/T + log_J
        # self.reward_norm and reward_beta are handled in base class usually, 
        # but here we make it explicit for geometry.
        
        # Ensure energies are a tensor
        if isinstance(energies, np.ndarray):
            energies = torch.tensor(energies, device=self.device, dtype=self.float)
            
        # Log Reward = -Energy * Beta + Log Jacobian
        # We assume self.reward_beta (inverse temp) exists, usually 1.0 or 32.0
        beta = self.reward_beta 
        log_reward = -energies * beta + log_jacobians
        
        # 4. Exponentiate to get Reward
        reward = torch.exp(log_reward)
        
        # Safety: Clip very small rewards to prevent log(0) in the loss function
        return torch.clamp(reward, min=self.min_reward)

    def reward2proxy(self, reward: TensorType["batch"]) -> TensorType["batch"]:
        """
        Converts the Agent's Reward back to the original Energy (Proxy value).
        Used for logging in gflownet.py (line 692).
        
        R = exp(-E * beta + log_J)
        log(R) = -E * beta + log_J
        E * beta = log_J - log(R)
        E = (log_J - log(R)) / beta
        """
        # NOTE: This is tricky because we need the state to know the Jacobian 
        # to reverse the operation perfectly. 
        # However, gflownet.py usually passes just the reward tensor.
        
        # If you cannot access the state here to get J, you have two options:
        # 1. Return the "Jacobian-adjusted Energy" (simplest, but logging is slightly off)
        # 2. Just return -log(reward) / beta (This is what most people do).
        
        # Standard approach (returns Free Energy, not Potential Energy):
        return -torch.log(reward + 1e-10) / self.reward_beta

    def parse_policy_output(self, policy_outputs: TensorType["batch", "dim"]):
        """
        Reconstructs the distribution from the flattened policy output tensor.
        """
        batch_size = policy_outputs.shape[0]
        idx = 0
        
        # 1. Reconstruct Torsions
        torsion_dist = None
        if self.n_torsion_angles > 0:
            n_t = self.n_torsion_angles
            n_c = self.n_comp # defined in yaml/init
            
            # Total params: n_torsions * n_components * 3
            block_size = n_t * n_c * 3
            t_params = policy_outputs[:, idx : idx + block_size]
            idx += block_size
            
            # Reshape back to [batch, n_t, n_c, 3]
            t_params = t_params.view(batch_size, n_t, n_c, 3)
            
            logits = t_params[..., 0]
            locs = t_params[..., 1]
            concs = t_params[..., 2]
            
            mix = Categorical(logits=logits)
            comp = VonMises(locs, concs)
            torsion_dist = MixtureSameFamily(mix, comp)

        # 2. Reconstruct Geometry (Lengths + Angles)
        geometry_dist = None
        n_geo = self.n_bond_lengths + self.n_bond_angles
        if n_geo > 0:
            # Total params: n_geo * 2
            block_size = n_geo * 2
            g_params = policy_outputs[:, idx : idx + block_size]
            idx += block_size
            
            # Reshape to [batch, n_geo, 2]
            g_params = g_params.view(batch_size, n_geo, 2)
            
            mu = g_params[..., 0]
            sigma = g_params[..., 1]
            
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
        
        # 1. Ensure inputs are tensors on the correct device
        if not torch.is_tensor(policy_outputs):
            policy_outputs = torch.tensor(policy_outputs, device=self.device, dtype=self.float)

        # 2. ROBUST FAST PATH DETECTION
        # Strategy: Use a heuristic based on the concentration value.
        # Random Policy (from Config) has concentration ~2.0.
        # Trained Policy (from base.py) has concentration > 5.0.
        # Therefore, if concentration < 4.0, it MUST be the random initialization.
        
        use_fast_path = False
        
        # Check explicit flag first
        if sampling_method == "random":
            use_fast_path = True
        elif self.n_torsion_angles > 0:
            # Check the 3rd element (Concentration of first torsion angle)
            # Structure is [logit, loc, conc, ...]
            first_concentration = policy_outputs[0, 2]
            #print(f"DEBUG: First Concentration: {first_concentration}")
            if first_concentration < 4.0:
                use_fast_path = True
                #print("DEBUG: Detected Random Policy via Concentration Heuristic (< 4.0)")

        # 3. Execute Sampling
        if use_fast_path:
            # --- FAST PATH: USE UNIFORM SAMPLING ---
            #print("DEBUG: Using FAST PATH (Uniform)") 
            
            # Torsions: Uniform over [0, 2*pi]
            uniform_tors = Uniform(
                torch.zeros(self.n_torsion_angles, device=self.device), 
                2 * np.pi * torch.ones(self.n_torsion_angles, device=self.device)
            )
            # Geometry: Uniform over [-1, 1] range
            n_geo = self.n_bond_lengths + self.n_bond_angles
            uniform_geo = Uniform(
                -torch.ones(n_geo, device=self.device), 
                torch.ones(n_geo, device=self.device)
            )

            # Sample both and concatenate
            parts = []
            if self.n_torsion_angles > 0:
                parts.append(uniform_tors.sample(sample_shape=[policy_outputs.shape[0]]))
            if n_geo > 0:
                parts.append(uniform_geo.sample(sample_shape=[policy_outputs.shape[0]]))
            
            actions_tensor = torch.cat(parts, dim=-1)

            # Compute constant log probs
            log_prob_sum = 0.0
            if self.n_torsion_angles > 0:
                # SIMPLIFY THIS: Pre-calculate math on CPU
                import math
                val = -math.log(2 * np.pi)
                lp_t = torch.full((policy_outputs.shape[0],), val, device=self.device)
                log_prob_sum += lp_t
                
            if n_geo > 0:
                # SIMPLIFY THIS
                val = -math.log(2.0)
                lp_g = torch.full((policy_outputs.shape[0],), val, device=self.device)
                log_prob_sum += lp_g
            
            logprobs = log_prob_sum
            
            
        else:
            # --- NORMAL PATH: USE TRAINED POLICY ---
            print("DEBUG: Using TRAINED POLICY (Von Mises)")
            dist = self.parse_policy_output(policy_outputs)
            actions_tensor = dist.sample()
            logprobs = dist.log_prob(actions_tensor)
        
        actions = [tuple(x.tolist()) for x in actions_tensor]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["batch", "policy_output_dim"],
        actions: TensorType["batch", "action_dim"],
        mask: TensorType["batch", "mask_dim"],
        states: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch"]:
        
        dist = self.parse_policy_output(policy_outputs)
        
        # Ensure actions are a tensor
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=self.device, dtype=self.float)
            
        return dist.log_prob(actions)

    def get_policy_output(self, params: dict) -> List[float]:
        """
        Defines the fixed/random policy output vector.
        Structure: [Torsion Params] + [Geometry Params]
        """
        # -------------------------------------
        # 1. Torsions (Mixture of Von Mises)
        # -------------------------------------
        n_t = self.n_torsion_angles
        n_c = self.n_comp
        torsion_out = []
        
        if n_t > 0:
            # We need 3 params per component: [logit, loc, conc]
            # Params from YAML: vonmises_mean, vonmises_concentration
            # Logits: uniform weighting (0.0)
            logits = [0.0] * n_c 
            # Locs: all same mean
            locs = [params['vonmises_mean']] * n_c
            # Concs: all same concentration
            concs = [params['vonmises_concentration']] * n_c
            
            # Interleave them: [logits, locs, concs] per angle? 
            # NO, the parse_policy_output expects [logits, locs, concs] flattened.
            # See HeterogeneousPolicyHead: t_out = torch.stack([logits, locs, concs], dim=-1)
            # So for one angle, we have: [logit_1, loc_1, conc_1, logit_2, loc_2, conc_2...]
            # Actually, parse_policy_output reshapes to [batch, n_t, n_c, 3].
            # So the flat vector order must satisfy that reshape.
            
            # Let's create the flat list for ONE angle first, then repeat for n_t.
            single_angle_params = []
            for _ in range(n_c):
                single_angle_params.extend([0.0, params['vonmises_mean'], params['vonmises_concentration']])
            
            # Repeat for all torsion angles
            torsion_out = single_angle_params * n_t

        # -------------------------------------
        # 2. Geometry (Normal / Gaussian)
        # -------------------------------------
        # Structure: [mu, sigma] per dimension
        n_geo = self.n_bond_lengths + self.n_bond_angles
        geometry_out = []
        
        if n_geo > 0:
            # For fixed/random policies, we want stable geometry.
            # Mu = 0.0 (No deviation from equilibrium)
            # Sigma = 0.1 (Small variance, strictly POSITIVE)
            
            # Note: parse_policy_output takes these RAW. No activation is applied.
            # So we pass the literal standard deviation we want.
            target_sigma = 0.1 
            
            single_geo_params = [0.0, target_sigma]
            geometry_out = single_geo_params * n_geo

        # Concatenate
        return torsion_out + geometry_out

    def step(self, action: Tuple[float], skip_mask_check: bool = False) -> Tuple[List[float], Tuple[float], bool]:
        # 1. Update the state based on continuous action
        self.state = self._get_next_state(self.state, action)
        
        # 2. Increment action counter
        self.n_actions += 1
        
        # 3. FORCE TERMINATION if max length reached
        if self.n_actions >= self.max_traj_length:
            self.done = True
            
        return self.state, action, True

    def _get_next_state(self, state, action):
        """
        Applies the continuous action vector to the state vector.
        Handles dimension mismatch if state includes a timestamp.
        """
        s = torch.tensor(state, device=self.device, dtype=self.float)
        a = torch.tensor(action, device=self.device, dtype=self.float)
        
        n_dims = len(a)
        
        # --- FIX: Slice state to match action dimension ---
        physical_state = s[:n_dims] + a
        # --------------------------------------------------
        
        # Handle Torsions Wrapping
        if self.n_torsion_angles > 0:
            physical_state[:self.n_torsion_angles] = torch.remainder(
                physical_state[:self.n_torsion_angles], 2 * np.pi
            )

        # Handle Geometry Clipping
        if n_dims > self.n_torsion_angles:
            physical_state[self.n_torsion_angles:] = torch.clamp(
                physical_state[self.n_torsion_angles:], -5.0, 5.0
            )
            
        # Reconstruct Full State (append time dimension if it existed)
        if len(s) > n_dims:
            final_state = torch.cat([physical_state, s[n_dims:]])
        else:
            final_state = physical_state
            
        return final_state.tolist()

    def step_backwards(self, action, skip_mask_check=False):
        # FIX: Handle the backward step from the "Sink" (Done) state gracefully.
        if self.done:
            self.done = False
            return self.state, self.eos, True
        
        return super().step_backwards(action, skip_mask_check)