#replace current energy proxy with an RDKit MMFF (Merck Molecular Force Field) calculation. 
# This is the standard way to get "ground truth" energy for small molecules including bond/angle terms.

from gflownet.proxy.base import Proxy
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

class MMFFEnergyProxy(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_reward = 1e-6 # Prevent log(0)

    def __call__(self, states):
        """
        states: List of states (which the environment can convert to RDKit mols)
        """
        energies = []
        
        # We need the environment to convert state -> molecule
        # Assuming you pass the environment wrapper or use the state conversion logic here
        # For this snippet, I assume 'states' are already RDKit Mols or Conformers
        # If they are raw vectors, you need to use env.set_conformer(state) first
        
        for mol in states:
            try:
                # 1. Ensure the molecule has the MMFF properties set up
                mp = AllChem.MMFFGetMoleculeProperties(mol)
                
                # 2. Get the Force Field object
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                
                # 3. Calculate Energy (includes Bonds, Angles, Torsions, VdW, Electrostatics)
                if ff:
                    energy = ff.CalcEnergy()
                else:
                    # Fallback if MMFF fails (rare for standard organic molecules)
                    energy = 1000.0 
            except Exception:
                energy = 1000.0
            
            energies.append(energy)
            
        return np.array(energies)