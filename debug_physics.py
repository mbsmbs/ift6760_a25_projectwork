import numpy as np
import torch
from gflownet.envs.conformers.conformer import Conformer
from rdkit.Chem import AllChem

def test_stability():
    # Known low-energy starting point (from previous debug)
    LOW_ENERGY_TORSIONS = [-1.70, -0.87] 
    
    # Tiny step in radians (0.1 degree)
    SMALL_STEP = 0.00174533 
    
    # 1. Initialize Environment at Valid Start
    env = Conformer(
        smiles="CC(=O)NC(C)C(=O)NC",
        n_torsion_angles=2,
        torsion_indices=[0, 1],
        remove_hs=True # Same setting as your training run
    )
    
    # Create the complete initial state (Torsions + Geometry + Time=0)
    initial_state = LOW_ENERGY_TORSIONS + [0.0] * (env.n_dim - 2) + [0.0]
    
    # Force environment to low-energy start (S_0)
    env.reset()
    env.state = initial_state
    
    print("--- S_0 (Initial State) ---")
    
    # Check energy at S_0 (should be low, ~33)
    mp_initial = AllChem.MMFFGetMoleculeProperties(env.conformer.rdk_mol, mmffVariant='MMFF94')
    ff_initial = AllChem.MMFFGetMoleculeForceField(env.conformer.rdk_mol, mp_initial)
    E_initial = ff_initial.CalcEnergy()
    print(f"Energy at S_0: {E_initial:.3f} kcal/mol (Target: ~33)")

    # 2. Define the Action (A_1)
    # Action: Small move on the first torsion, 0 on the second
    action = [SMALL_STEP, 0.0] + [0.0] * (env.n_dim - 2)
    
    # 3. Take the Step (S_0 -> S_1)
    print("\n--- Taking Step A_1 ---")
    final_state, _, done = env.step(action)
    
    # 4. Check Energy at S_1
    env.sync_conformer_with_state(final_state)
    
    mp_final = AllChem.MMFFGetMoleculeProperties(env.conformer.rdk_mol, mmffVariant='MMFF94')
    
    if mp_final is None:
        print("❌ CRITICAL FAILURE: MMFF Properties failed after a tiny step.")
        return

    ff_final = AllChem.MMFFGetMoleculeForceField(env.conformer.rdk_mol, mp_final)
    E_final = ff_final.CalcEnergy()
    
    print(f"Energy at S_1: {E_final:.3f} kcal/mol")
    
    # 5. Determine Result
    if np.abs(E_final - E_initial) < 1.0:
        print("✅ SUCCESS: The energy is stable. The issue is purely the policy's exploration size.")
    elif E_final > 100.0:
        print("❌ CRITICAL FAILURE: A tiny step broke the molecule. The issue is likely RDKit/Conformer class instability.")
        
if __name__ == "__main__":
    test_stability()