import torch
from simulation.env import PodRacerEnv, STAGE_DUEL_FUSED
from config import *

def test_zone_pressure():
    print("=== Testing Zone Pressure Reward ===")
    
    # Setup
    env = PodRacerEnv(num_envs=1, device='cpu', start_stage=STAGE_DUEL_FUSED)
    env.reset()
    
    # Configure for Manual Test
    env.curriculum_stage = STAGE_DUEL_FUSED
    env.config.use_bots = False # We control positions manually
    
    # Force Pod 0 to be Blocker, Pod 2 to be Runner (Target)
    env.is_runner[0, 0] = False # Blocker
    env.is_runner[0, 2] = True # Runner
    
    # Setup Weights: Enable Pressure, Disable others for clarity
    weights = torch.zeros(1, 20)
    weights[0, RW_ZONE_PRESSURE] = 1.0 
    
    # Scenario 1: Perfect Alignment (Blocker between Runner and CP)
    # CP at (1000, 0)
    # Runner at (0, 0)
    # Blocker at (500, 0) -> Alignment should be 1.0
    
    env.checkpoints[0, 1] = torch.tensor([1000.0, 0.0]) # Next CP
    env.next_cp_id[0, 2] = 1 # Runner aiming at CP 1
    
    env.physics.pos[0, 2] = torch.tensor([0.0, 0.0]) # Runner
    env.physics.vel[0, 2] = torch.tensor([100.0, 0.0])
    
    env.physics.pos[0, 0] = torch.tensor([500.0, 0.0]) # Blocker
    env.physics.vel[0, 0] = torch.tensor([0.0, 0.0])
    
    # Step
    rewards, _, _ = env.step(torch.zeros(1, 4, 4), reward_weights=weights)
    
    # Check Reward
    # Reward = Align(1.0) * Risk(factor) * Weight(1.0)
    # Dist En->CP = 1000. Risk = 1.0 - (1000/5000) = 0.8
    # Exp Reward ~ 0.8
    
    rew_1 = rewards[0, 0].item()
    print(f"Scenario 1 (Perfect Block): Reward {rew_1:.4f} (Expected ~0.8)")
    
    # Scenario 2: Bad Alignment (Blocker behind Runner)
    # Runner at (0, 0)
    # Blocker at (-500, 0)
    # env.rewards.zero_() # No need, we use returned val
    env.physics.pos[0, 0] = torch.tensor([-500.0, 0.0])
    rewards, _, _ = env.step(torch.zeros(1, 4, 4), reward_weights=weights)
    rew_2 = rewards[0, 0].item()
    print(f"Scenario 2 (Behind Runner): Reward {rew_2:.4f} (Expected 0.0)")
    
    # Scenario 3: Perpendicular (Side)
    # Blocker at (500, 500)
    # Vec Opp->Me = (500, 500). Dir = (0.707, 0.707)
    # Vec Opp->CP = (1000, 0). Dir = (1.0, 0.0)
    # Dot = 0.707
    env.physics.pos[0, 0] = torch.tensor([500.0, 500.0])
    rewards, _, _ = env.step(torch.zeros(1, 4, 4), reward_weights=weights)
    rew_3 = rewards[0, 0].item()
    print(f"Scenario 3 (Side Block): Reward {rew_3:.4f} (Expected ~{0.707 * 0.8:.4f})")
    
    if rew_1 > 0.7 and rew_2 == 0.0 and rew_3 > 0.0:
        print(">>> TEST PASSED <<<")
    else:
        print(">>> TEST FAILED <<<")

if __name__ == "__main__":
    test_zone_pressure()
