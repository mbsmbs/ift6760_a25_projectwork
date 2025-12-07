import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints, rdMolTransforms
from rdkit.Geometry.rdGeometry import Point3D

from gflownet.utils.molecule import constants

def get_torsion_angles_atoms_list(mol):
    return [x[0][0] for x in TorsionFingerprints.CalculateTorsionLists(mol)[0]]

def get_torsion_angles_values(conf, torsion_angles_atoms_list):
    return [
        np.float32(rdMolTransforms.GetDihedralRad(conf, *ta))
        for ta in torsion_angles_atoms_list
    ]

def get_all_torsion_angles(mol, conf):
    ta_atoms = get_torsion_angles_atoms_list(mol)
    ta_values = get_torsion_angles_values(conf, ta_atoms)
    return {k: v for k, v in zip(ta_atoms, ta_values)}

class RDKitConformer:
    def __init__(self, atom_positions, smiles, freely_rotatable_tas=None, 
                 freely_variable_bonds=None, freely_variable_angles=None):
        
        self.smiles = smiles
        self.rdk_mol = self.get_mol_from_smiles(smiles)
        
        # --- OPTIMIZATION: INSTANT INITIALIZATION ---
        # We DO NOT call EmbedMolecule here. It is slow (100ms+).
        # Instead, we create a blank conformer container.
        print("Creating blank conformer container")
        conf = Chem.Conformer(self.rdk_mol.GetNumAtoms())
        self.rdk_mol.AddConformer(conf)
        self.rdk_conf = self.rdk_mol.GetConformer()
        # --------------------------------------------

        # Set the positions immediately
        self.set_atom_positions(atom_positions)

        self.freely_rotatable_tas = freely_rotatable_tas if freely_rotatable_tas else []
        self.bond_length_pairs = freely_variable_bonds if freely_variable_bonds else []
        self.bond_angle_triplets = freely_variable_angles if freely_variable_angles else []

    def __deepcopy__(self, memo):
        print("Deepcopying RDKitConformer")
        cls = self.__class__
        new_obj = cls.__new__(cls)
        
        # 1. Copy Attributes
        new_obj.smiles = self.smiles
        new_obj.freely_rotatable_tas = self.freely_rotatable_tas
        new_obj.bond_length_pairs = self.bond_length_pairs
        new_obj.bond_angle_triplets = self.bond_angle_triplets
        
        # --- OPTIMIZATION: INSTANT COPY ---
        # Clone the RDKit object in memory. This preserves the conformer 
        # and positions instantly without calculation.
        new_obj.rdk_mol = Chem.Mol(self.rdk_mol)
        new_obj.rdk_conf = new_obj.rdk_mol.GetConformer()
        # ----------------------------------
        
        # Sync positions safety check
        if hasattr(self, 'get_atom_positions'):
            new_obj.set_atom_positions(self.get_atom_positions())
            
        return new_obj

    def get_mol_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        return mol

    def set_atom_positions(self, atom_positions):
        for idx, pos in enumerate(atom_positions):
            self.rdk_conf.SetAtomPosition(idx, Point3D(*pos))

    def get_atom_positions(self):
        return self.rdk_conf.GetPositions()

    def get_atomic_numbers(self):
        atomic_numbers = [atom.GetAtomicNum() for atom in self.rdk_mol.GetAtoms()]
        return np.array(atomic_numbers)

    def get_n_atoms(self):
        return self.rdk_mol.GetNumAtoms()

    # --- TORSION HELPERS ---
    def set_torsion_angle(self, torsion_angle, value):
        rdMolTransforms.SetDihedralRad(self.rdk_conf, *torsion_angle, float(value))

    def set_torsion_vector(self, torsions, values):
        for (i, j, k, l), v in zip(torsions, values):
            rdMolTransforms.SetDihedralRad(self.rdk_conf, i, j, k, l, float(v))

    def get_freely_rotatable_tas_values(self):
        return get_torsion_angles_values(self.rdk_conf, self.freely_rotatable_tas)

    # --- BOND LENGTH HELPERS ---
    def get_bond_length(self, i, j):
        return rdMolTransforms.GetBondLength(self.rdk_conf, int(i), int(j))

    def set_bond_length(self, i, j, value):
        rdMolTransforms.SetBondLength(self.rdk_conf, int(i), int(j), float(value))

    def get_bond_length_vector(self, bond_pairs):
        lengths = []
        for i, j in bond_pairs:
            lengths.append(rdMolTransforms.GetBondLength(self.rdk_conf, i, j))
        return np.array(lengths, dtype=float)

    def set_bond_length_vector(self, bond_pairs, values):
        for (i, j), v in zip(bond_pairs, values):
            rdMolTransforms.SetBondLength(self.rdk_conf, i, j, float(v))
            
    def get_bond_lengths_values(self):
        return [self.get_bond_length(i, j) for (i, j) in self.bond_length_pairs]

    # --- BOND ANGLE HELPERS ---
    def get_bond_angle(self, i, j, k):
        return rdMolTransforms.GetAngleRad(self.rdk_conf, int(i), int(j), int(k))

    def set_bond_angle(self, i, j, k, value):
        rdMolTransforms.SetAngleRad(self.rdk_conf, int(i), int(j), int(k), float(value))

    def get_angle_vector(self, angle_triplets):
        angles = []
        for i, j, k in angle_triplets:
            angles.append(rdMolTransforms.GetAngleRad(self.rdk_conf, i, j, k))
        return np.array(angles, dtype=float)

    def set_angle_vector(self, angle_triplets, values):
        for (i, j, k), v in zip(angle_triplets, values):
            rdMolTransforms.SetAngleRad(self.rdk_conf, i, j, k, float(v))

    def get_bond_angles_values(self):
        return [self.get_bond_angle(i, j, k) for (i, j, k) in self.bond_angle_triplets]