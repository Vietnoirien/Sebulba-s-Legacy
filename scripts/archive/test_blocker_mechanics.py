import torch
import numpy as np
from simulation.env import PodRacerEnv
from config import TrainingConfig, STAGE_DUEL_FUSED

def test_blocker_mechanics():
    print("=== Testing Blocker Mechanics (Front Intercept & Velocity Denial) ===")
    
    cfg = TrainingConfig()
    cfg.num_envs = 4 # Small batch
    cfg.device = 'cpu'
    
    env = PodRacerEnv(4, device='cpu')
    env.curriculum_stage = STAGE_DUEL_FUSED # Force Stage 2
    
    # Reset
    env.reset()
    
    # ISOLATION: Disable Intercept Reward to verify Denial Reward pure signal
    env.reward_scaling_config.intercept_progress_scale = 0.0
    env.reward_scaling_config.velocity_denial_weight = 100.0 # Ensure this is set
    
    # Setup Scenario:
    # Pod 0: Blocker (Me)
    # Pod 1: Teammate
    # Pod 2: Enemy 1
    # Pod 3: Enemy 2
    
    # 1. Force Roles
    env.is_runner[:] = False # All Blockers?
    env.is_runner[:, 0] = False # Blocker
    env.is_runner[:, 2] = True # Enemy Runner
    env.is_runner[:, 3] = True # Enemy Runner
    
    # 2. Setup Positions
    # CP is at (1000, 1000)
    # Enemy is at (0, 0), moving towards CP
    # Me is at (0, 100)
    
    cp_pos = env.checkpoints[:, 1] # Next CP
    env.next_cp_id[:, 2] = 1 # Enemy Next CP
    env.next_cp_id[:, 3] = 1 
    
    # Case A: Enemy Moving Fast to CP -> High Penalty (Denial Logic)
    # Enemy Vel = Towards CP * 800
    dir_to_cp = cp_pos[0] - env.physics.pos[0, 2] # Use relative to Pod 2
    dir_to_cp = dir_to_cp / torch.norm(dir_to_cp)
    
    # Set BOTH enemies to fast velocity so target selection doesn't matter
    env.physics.vel[0, 2] = dir_to_cp * 800.0
    env.physics.vel[0, 3] = dir_to_cp * 800.0
    
    # Me: Stationary
    env.physics.pos[0, 0] = torch.tensor([0.0, 100.0])
    env.physics.vel[0, 0] = torch.zeros(2)
    
    # Teammate: Move FAR away to avoid collision penalty
    env.physics.pos[0, 1] = torch.tensor([5000.0, 5000.0])
    env.physics.vel[0, 1] = torch.zeros(2)
    
    # Create dummy reward weights
    reward_weights = torch.ones((4, 25), device='cpu') 
    
    # Step
    ret = env.step(torch.zeros(4, 4, 4), reward_weights)
    if len(ret) == 3: rew, _, _ = ret
    elif len(ret) == 4: _, rew, _, _ = ret
    else: raise ValueError(f"Env returned {len(ret)} values")
    
    # Parse Reward for Pod 0
    r_denial_fast = rew[0, 0].item()
    print(f"Case A (Enemy Fast to CP): Blocker Reward = {r_denial_fast:.2f}")
    
    # Case B: Enemy Moving Away (Pushed Back) -> Positive Reward
    env.is_runner[:] = False # [FIX] Re-force roles as step() resets them
    env.physics.vel[0, 2] = -dir_to_cp * 200.0 # Backwards
    env.physics.vel[0, 3] = -dir_to_cp * 200.0 
    
    ret = env.step(torch.zeros(4, 4, 4), reward_weights)
    if len(ret) == 3: rew, _, _ = ret
    elif len(ret) == 4: _, rew, _, _ = ret
        
    r_denial_push = rew[0, 0].item()
    print(f"Case B (Enemy Pushed Back): Blocker Reward = {r_denial_push:.2f}")
    
    # Case C: Intercept Logic
    me_p = torch.tensor([[0.0, 0.0]])
    en_p = torch.tensor([[1000.0, 0.0]])
    en_v = torch.tensor([[100.0, 0.0]])
    
    pred = env._get_front_intercept_pos(me_p, en_p, en_v)
    print(f"Intercept Prediction: En={en_p[0].numpy()} Vel={en_v[0].numpy()} -> Pred={pred[0].numpy()}")
    
if __name__ == "__main__":
    test_blocker_mechanics()
