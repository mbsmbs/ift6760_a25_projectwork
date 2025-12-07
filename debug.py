import torch
import numpy as np
import logging
import sys

# Attempt imports from your codebase
try:
    from gflownet.envs.conformers.conformer import Conformer
    from gflownet.policy.base import HeterogeneousPolicyHead
    from gflownet.proxy.molecule import MMFFEnergyProxy
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please run this script from the root of your project directory.")
    sys.exit(1)

# Configuration for the test (Simulating your YAML)
IBUPROFEN_SMILES = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
CONFIG = {
    'smiles': IBUPROFEN_SMILES,
    'n_torsion_angles': -1, # Auto-detect
    'flex_bond_lengths': [[0, 1], [1, 2], [1, 3], [3, 4], [7, 10], [10, 11], [10, 12], [12, 14]],
    'flex_bond_angles': [[0, 1, 2], [0, 1, 3], [2, 1, 3], [1, 3, 4], [3, 4, 5], [7, 10, 11], [7, 10, 12], [10, 12, 14]],
    'length_scale': 0.1,
    'angle_scale': 0.4,
    'reward_beta': 32.0,
    # Policy params
    'n_hid': 64,
    'n_comp': 3,
}

def test_gradient_flow():
    print("\n=== TEST 1: Policy Gradient Flow ===")
    
    # 1. Setup Dummy Dimensions based on Config
    n_torsions = 4 # Approx for Ibuprofen
    n_lengths = len(CONFIG['flex_bond_lengths'])
    n_angles = len(CONFIG['flex_bond_angles'])
    backbone_dim = CONFIG['n_hid']
    
    print(f"Dimensions: Torsions={n_torsions}, Lengths={n_lengths}, Angles={n_angles}")

    # 2. Instantiate the Head
    policy = HeterogeneousPolicyHead(
        backbone_dim=backbone_dim,
        n_torsions=n_torsions,
        n_lengths_angles=n_lengths + n_angles,
        n_components=CONFIG['n_comp']
    )
    
    # 3. Create Dummy Input (Embedding)
    batch_size = 5
    embedding = torch.randn(batch_size, backbone_dim, requires_grad=True)
    
    # 4. Forward Pass
    dist = policy(embedding)
    print("Forward pass successful.")
    
    # 5. Sample Action
    # This simulates the agent picking a move
    actions = dist.sample()
    print(f"Sampled Action Shape: {actions.shape} (Expected: [{batch_size}, {n_torsions + n_lengths + n_angles}])")
    
    # 6. Compute Log Prob
    # This is the crucial step for RL/GFN loss
    log_probs = dist.log_prob(actions)
    print(f"Log Prob Shape: {log_probs.shape}")
    
    # 7. Backward Pass (Simulate Loss)
    # We want to maximize log_prob (or minimize negative log_prob)
    loss = -log_probs.mean()
    loss.backward()
    
    # 8. Check Gradients
    print("\nChecking Gradients...")
    if policy.torsion_out.weight.grad is not None:
        grad_norm = policy.torsion_out.weight.grad.norm().item()
        print(f"✔ Torsion Head Gradient Norm: {grad_norm:.6f}")
    else:
        print("❌ Torsion Head has NO gradient!")

    if policy.geometry_out.weight.grad is not None:
        grad_norm = policy.geometry_out.weight.grad.norm().item()
        print(f"✔ Geometry Head Gradient Norm: {grad_norm:.6f}")
    else:
        print("❌ Geometry Head has NO gradient!")
        
    if embedding.grad is not None:
        print(f"✔ Backbone Embedding Gradient Norm: {embedding.grad.norm().item()}")
    else:
        print("❌ Backbone has NO gradient!")


def test_physics_engine():
    print("\n=== TEST 2: Physics Engine (NeRF + Jacobian + Reward) ===")
    
    # 1. Instantiate Environment
    # We pass kwargs to match what we put in the YAML
    env = Conformer(
        smiles=CONFIG['smiles'],
        n_torsion_angles=CONFIG['n_torsion_angles'],
        flex_bond_lengths=CONFIG['flex_bond_lengths'],
        flex_bond_angles=CONFIG['flex_bond_angles'],
        length_scale=CONFIG['length_scale'],
        angle_scale=CONFIG['angle_scale'],
        reward_beta=CONFIG['reward_beta']
    )
    
    print(f"Environment initialized for: {env.smiles}")
    print(f"Detected {env.n_torsion_angles} Torsions, {env.n_bond_lengths} Lengths, {env.n_bond_angles} Angles.")
    
    # 2. Get Initial State
    # Conformer states usually include 'time', so we might need to handle that.
    # We'll just generate a random valid action vector.
    total_dims = env.n_torsion_angles + env.n_bond_lengths + env.n_bond_angles
    
    # Create a batch of random "intrinsic coordinate actions"
    # Small random noise around 0 (equilibrium)
    dummy_actions = np.random.normal(0, 0.1, size=(3, total_dims)).tolist()
    
    # 3. Test Jacobian Calculation
    print("\nChecking Jacobian...")
    for i, action in enumerate(dummy_actions):
        log_det_j = env.get_log_jacobian(action)
        print(f"Sample {i}: Log Jacobian = {log_det_j:.4f}")
        
        if np.isinf(log_det_j) or np.isnan(log_det_j):
            print("❌ Jacobian is Infinite or NaN! (Check for zero bond lengths)")
        else:
            print("✔ Jacobian is finite.")

    # 4. Test Reward (MMFF Energy)
    print("\nChecking MMFF Proxy...")
    try:
        proxy = MMFFEnergyProxy(device='cpu', float_precision=32)
        
        # We need to convert actions to states that the proxy understands
        # In GFlowNet, env.state2proxy or similar usually handles this.
        # Here we manually sync to get the RDKit mol.
        mols = []
        for action in dummy_actions:
            env.sync_conformer_with_state(action)
            # Make a copy to ensure we don't overwrite
            mols.append(env.conformer.rdk_mol)
            
        # Compute Energy
        energies = proxy(mols)
        print(f"Energies (kcal/mol): {energies}")
        
        # 5. Compute Final Reward
        # Reward = exp( -beta * Energy + log_J )
        beta = CONFIG['reward_beta']
        for j, (energy, action) in enumerate(zip(energies, dummy_actions)):
            log_jac = env.get_log_jacobian(action)
            log_reward = -beta * energy + log_jac
            reward = np.exp(log_reward)
            print(f"Sample {j}: Final Reward = {reward:.6e}")
            
    except Exception as e:
        print(f"❌ Reward Calculation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradient_flow()
    test_physics_engine()