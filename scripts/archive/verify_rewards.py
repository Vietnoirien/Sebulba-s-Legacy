
import torch
import numpy as np
from simulation.env import PodRacerEnv, STAGE_DUEL_FUSED
from config import CurriculumConfig, EnvConfig, RewardScalingConfig, DEFAULT_REWARD_WEIGHTS

def run_verification():
    print("--- Verifying Reward Fixes ---")
    
    # 1. Setup Environment in Stage 2 (Duel) where rewards are active
    curr_config = CurriculumConfig()
    env_config = EnvConfig(
        mode_name="duel",
        active_pods=[0, 1, 2, 3],
        use_bots=False 
    )
    
    num_envs = 1
    env = PodRacerEnv(num_envs, device="cpu", start_stage=STAGE_DUEL_FUSED)
    env.config = env_config
    env.curriculum_config = curr_config
    env.reset()
    
    # Ensure Pod 0 is Blocker (RoleID 1) and Pod 2 is Runner (RoleID 0)
    # env.is_runner is [B, 4]. 
    # Let's force roles for Env 0.
    # Pod 0: Blocker (False)
    # Pod 2: Runner (True)
    env.is_runner[0, 0] = False 
    env.is_runner[0, 2] = True
    
    print(f"Pod 0 Role: {'Runner' if env.is_runner[0,0] else 'Blocker'}")
    print(f"Pod 2 Role: {'Runner' if env.is_runner[0,2] else 'Blocker'}")
    
    # --- Test 1: Intercept Reward Scaling ---
    print("\n[Test 1] Intercept Reward (Pod 0 moving towards intercept point)")
    # Scenario: Enemy (Pod 2) is stationary at [10000, 5000].
    # Me (Pod 0) is at [8000, 5000].
    # Intercept Point should be roughly Enemy Position (since Enemy Vel is 0).
    # Me moving towards Enemy at various speeds.
    
    test_speeds = [0.0, 200.0, 400.0, 800.0]
    
    for speed in test_speeds:
        # Reset state
        env.physics.pos[0, 0] = torch.tensor([8000.0, 5000.0])
        env.physics.vel[0, 0] = torch.tensor([speed, 0.0]) # Moving East towards Enemy
        
        env.physics.pos[0, 2] = torch.tensor([10000.0, 5000.0])
        env.physics.vel[0, 2] = torch.tensor([0.0, 0.0]) # Stationary
        
        # Zero out other pods to avoid noise
        env.physics.pos[0, 1] = torch.tensor([-1000.0, -1000.0])
        env.physics.pos[0, 3] = torch.tensor([-2000.0, -2000.0])
        
        # Capture metrics before step? Metrics are accumulated in step.
        # We need to peek at 'blocker_intercept_metric' returned in infos.
        
        # Step
        # Prepare weights
        weights = torch.zeros((num_envs, 20))
        for k, v in DEFAULT_REWARD_WEIGHTS.items():
             weights[:, k] = v
        
        _, _, infos = env.step(torch.zeros((env.num_envs, 4, 4)), weights)
        
        # Check Intercept Metric for Pod 0
        intercept_rew = infos['blocker_intercept'][0, 0].item()
        print(f"  Speed {speed:5.1f} -> Intercept Reward: {intercept_rew:6.2f} (Target at Max: ~40.0)")

    # --- Test 2: Velocity Denial Penalty ---
    print("\n[Test 2] Velocity Denial Penalty (Enemy moves, Me stationary)")
    # Scenario: Me (Pod 0) at [8000, 5000], Stationary.
    # Enemy (Pod 2) at [10000, 5000], Moving towards Checkpoint (assume CP is East).
    # We need to make sure Enemy is moving towards next CP.
    # We'll set Enemy next CP to be further East.
    
    # Force Next CP
    env.next_cp_id[0, 2] = 1
    env.checkpoints[0, 1] = torch.tensor([15000.0, 5000.0]) # CP East
    
    enemy_speeds = [0.0, 5.0, 15.0, 60.0, 100.0]
    
    for e_speed in enemy_speeds:
        # Reset Logic for cleanliness (simpler than full reset)
        env.physics.pos[0, 0] = torch.tensor([8000.0, 5000.0])
        env.physics.vel[0, 0] = torch.tensor([0.0, 0.0]) # Stationary
        
        env.physics.pos[0, 2] = torch.tensor([10000.0, 5000.0])
        env.physics.vel[0, 2] = torch.tensor([e_speed, 0.0]) # Moving East (Towards CP)
        
        # Reset metrics manually to read just this step
        # Actually infos returns current step metrics for these specific ones (constructed in step)
        
        _, _, infos = env.step(torch.zeros((env.num_envs, 4, 4)), weights)
        
        # Check Blocker Damage Metric (which accumulates Denial Reward)
        # Note: If no collision, this IS the Denial Reward.
        # Ensure no collision: Distance 2000.0, Radius 400+400=800. Safe.
        
        denial_val = infos['blocker_damage'][0, 0].item()
        print(f"  Enemy Speed {e_speed:5.1f} -> Denial Reward: {denial_val:6.2f} (Target: -10.0 if > 10.0)")

if __name__ == "__main__":
    with torch.no_grad():
        run_verification()
