
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from simulation.env import PodRacerEnv, STAGE_DUEL_FUSED, RW_COLLISION_BLOCKER, RW_CHECKPOINT
from config import EnvConfig

def test_rewards():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup Env
    env = PodRacerEnv(num_envs=7, device=device, start_stage=STAGE_DUEL_FUSED)
    
    # Scenario 6: Escorting (Parallel low reward)
    # Scenario 7: Intercepting (Positional high reward)
    
    # Environment indices
    env_passive = 0
    env_active = 1
    env_fail = 2
    env_helping = 3
    env_pinning = 4
    env_escort = 5
    env_intercept = 6
    
    env.reset()
    
    # Force roles
    env.is_runner[:] = False 
    env.is_runner[:, 0] = False
    env.is_runner[:, 2] = True
    
    # --- Scenario 1-5 (Same as before) ---
    # ... (Please assume previous setup code is preserved/re-applied if I were rewriting, 
    # but I am using Replace, so I must match existing context or append.)
    # I will just reconstruct the updated setup block for clarity or assume you want me to append?
    # I will replace the setup block.
    
    # --- Scenario 1: Passive ---
    env.physics.pos[env_passive, 0] = torch.tensor([1000.0, 1000.0], device=device)
    env.physics.vel[env_passive, 0] = torch.tensor([0.0, 0.0], device=device)
    env.physics.pos[env_passive, 2] = torch.tensor([1000.0, 500.0], device=device)
    env.physics.vel[env_passive, 2] = torch.tensor([0.0, 800.0], device=device)
    env.next_cp_id[env_passive, 2] = 1
    env.checkpoints[env_passive, 1] = torch.tensor([1000.0, 5000.0], device=device)
    
    # --- Scenario 2: Active ---
    env.physics.pos[env_active, 0] = torch.tensor([1000.0, 1500.0], device=device)
    env.physics.vel[env_active, 0] = torch.tensor([0.0, -800.0], device=device)
    env.physics.pos[env_active, 2] = torch.tensor([1000.0, 1000.0], device=device)
    env.physics.vel[env_active, 2] = torch.tensor([0.0, 0.0], device=device)
    env.next_cp_id[env_active, 2] = 1
    env.checkpoints[env_active, 1] = torch.tensor([1000.0, 5000.0], device=device)

    # --- Scenario 3: Failed ---
    env.physics.pos[env_fail, 0] = torch.tensor([1000.0, 1000.0], device=device)
    env.physics.vel[env_fail, 0] = torch.tensor([0.0, 0.0], device=device)
    env.physics.pos[env_fail, 2] = torch.tensor([1000.0, 500.0], device=device)
    env.physics.vel[env_fail, 2] = torch.tensor([0.0, 800.0], device=device)
    env.next_cp_id[env_fail, 2] = 1
    env.checkpoints[env_fail, 1] = torch.tensor([1000.0, 900.0], device=device)

    # --- Scenario 4: Helping ---
    env.physics.pos[env_helping, 0] = torch.tensor([1000.0, 500.0], device=device)
    env.physics.vel[env_helping, 0] = torch.tensor([0.0, 800.0], device=device)
    env.physics.pos[env_helping, 2] = torch.tensor([1000.0, 1000.0], device=device)
    env.physics.vel[env_helping, 2] = torch.tensor([0.0, 0.0], device=device)
    env.next_cp_id[env_helping, 2] = 1
    env.checkpoints[env_helping, 1] = torch.tensor([1000.0, 5000.0], device=device)

    # --- Scenario 5: Pinning ---
    env.physics.pos[env_pinning, 0] = torch.tensor([1000.0, 605.0], device=device)
    env.physics.vel[env_pinning, 0] = torch.tensor([0.0, 150.0], device=device)
    env.physics.pos[env_pinning, 2] = torch.tensor([1000.0, 1000.0], device=device)
    env.physics.vel[env_pinning, 2] = torch.tensor([0.0, 100.0], device=device)
    env.next_cp_id[env_pinning, 2] = 1
    env.checkpoints[env_pinning, 1] = torch.tensor([1000.0, 5000.0], device=device)
    
    # --- Scenario 6: Escorting (Parallel) ---
    # Runner moving North (0, 800) at 1000, 1000.
    # Blocker moving North (0, 800) at 1400, 1000 (Side-by-side, dist 400 - Collision)
    # Alignment = 1.0. Position Bias? Side (0).
    env.physics.pos[env_escort, 0] = torch.tensor([1400.0, 1000.0], device=device)
    env.physics.vel[env_escort, 0] = torch.tensor([0.0, 800.0], device=device)
    env.physics.pos[env_escort, 2] = torch.tensor([1000.0, 1000.0], device=device)
    env.physics.vel[env_escort, 2] = torch.tensor([0.0, 800.0], device=device)
    env.next_cp_id[env_escort, 2] = 1
    env.checkpoints[env_escort, 1] = torch.tensor([1000.0, 5000.0], device=device)
    
    # --- Scenario 7: Intercepting (Static Zone) ---
    # Runner moving North (0, 800) at 1000, 1000.
    # Blocker Stationary at 1000, 2000 (In Front).
    # Alignment = 0 (v_b=0). Position Bias = In Front.
    env.physics.pos[env_intercept, 0] = torch.tensor([1000.0, 2000.0], device=device)
    env.physics.vel[env_intercept, 0] = torch.tensor([0.0, 0.0], device=device)
    env.physics.pos[env_intercept, 2] = torch.tensor([1000.0, 1000.0], device=device)
    env.physics.vel[env_intercept, 2] = torch.tensor([0.0, 800.0], device=device)
    env.next_cp_id[env_intercept, 2] = 1
    env.checkpoints[env_intercept, 1] = torch.tensor([1000.0, 5000.0], device=device)
    
    # --- STEP ---
    # No actions
    actions = torch.zeros((7, 4, 4), device=device)
    
    # Weights: Collision=1, Denial=1 (to test Zone)
    weights = torch.zeros((7, 16), device=device)
    weights[:, RW_COLLISION_BLOCKER] = 1.0 
    weights[:, 15] = 10000.0 # RW_DENIAL (Check index in env.py, usually 15)
                             # Enforce large weight to see effect
    
    # Step
    print("Stepping...")
    rewards, dones, infos = env.step(actions, weights, tau=0.0)
    
    # --- ANALYZE REWARDS ---
    
    print("\n--- RESULTS ---")
    
    r_passive = rewards[env_passive, 0].item()
    print(f"Scenario 1 (Passive): Reward = {r_passive:.2f}")
    
    r_active = rewards[env_active, 0].item()
    print(f"Scenario 2 (Active Correct): Reward = {r_active:.2f}")
    
    r_fail = rewards[env_fail, 0].item()
    print(f"Scenario 3 (Failed Block): Reward = {r_fail:.2f}")
    
    r_helping = rewards[env_helping, 0].item()
    print(f"Scenario 4 (Helping): Reward = {r_helping:.2f}")
    
    r_pinning = rewards[env_pinning, 0].item()
    print(f"Scenario 5 (Pinning): Reward = {r_pinning:.2f}")
    
    r_escort = rewards[env_escort, 0].item()
    print(f"Scenario 6 (Escort): Reward = {r_escort:.2f} (Expect Low/Zero)")
    
    r_intercept = rewards[env_intercept, 0].item()
    print(f"Scenario 7 (Intercept): Reward = {r_intercept:.2f} (Expect High Zone)")

if __name__ == "__main__":
    try:
        test_rewards()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
