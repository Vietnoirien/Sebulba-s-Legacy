
import torch
import sys
import os
import math

# Add root to path
sys.path.append(os.getcwd())

from simulation.env import PodRacerEnv, STAGE_TEAM
from config import TrainingConfig, EnvConfig

def verify_bot_braking():
    print("--- Verifying Bot Braking (Parking) Logic ---")
    
    # 1. Setup Env
    cfg = TrainingConfig()
    env_cfg = EnvConfig(active_pods=[0,1,2,3], use_bots=True, bot_pods=[2,3])
    env = PodRacerEnv(num_envs=1, device='cpu', start_stage=STAGE_TEAM)
    env.set_stage(STAGE_TEAM, env_cfg, reset_env=True)
    env.bot_difficulty = 1.0
    
    # Force Bot 2 to be Blocker (is_runner = False)
    # is_runner is [Batch, 4] boolean
    env.is_runner[:, 2] = False
    
    # 2. Scenario: Bot arriving at Gatekeeper Point
    env.reset()
    
    # Pod 2 (Bot)
    # Runner (0) far away. 
    # CP at (10000, 0).
    # Bot moving towards CP. 
    # We want to check if thrust goes to 0 when close.
    
    cp_idx = env.next_cp_id[0, 0]
    cp_pos = env.checkpoints[0, cp_idx]
    
    # Bot Pos: 1000 units from CP.
    # Gatekeeper offset is roughly 500 (since diff=1.0).
    # So Bot is roughly 500 units from Gatekeeper Point. 
    # Wait, Gatekeeper Point = CP - Offset.
    # Offset = 500.
    # Gate Point = CP - 500.
    # If Bot is at CP - 1000, dist to Gate is 500.
    # Dist < 600 -> Should be 0 Thrust.
    
    gate_pos_est = cp_pos - torch.tensor([500.0, 0.0]) # Approx
    
    # Place Bot at gate_pos_est + small error
    start_pos = gate_pos_est - torch.tensor([400.0, 0.0]) # Dist ~ 400
    env.physics.pos[:, 2] = start_pos
    
    # Moving towards gate
    env.physics.vel[:, 2] = torch.tensor([200.0, 0.0]) # Significant speed
    env.physics.angle[:, 2] = 0.0 # Facing Gate
    
    # Ensure it chooses Gatekeep logic
    # Runner (0) needs to be close enough to trigger Red Zone (<4000) logic, 
    # BUT far enough (>800) from Bot to avoid Emergency Ram.
    
    # Bot at Gate roughly (CP-500).
    # Actual Bot Pos set to Gate - 400 (CP-900).
    # Runner needs to be > 800 away to avoid Emergency Ram.
    # Place Runner at CP-2200.
    # Dist Bot-Runner = 1300. (>800).
    # Dist Runner-CP = 2200. (<4000). Force Gatekeep.
    
    env.physics.pos[:, 0] = cp_pos - torch.tensor([2200.0, 0.0]) 
    
    # Also force difficulty high to enable logic scaling
    env.bot_difficulty = 1.0
    
    # Step
    actions = torch.zeros((1, 4, 4))
    
    # We need to capture the thrust applied.
    # It's not exposed directly.
    # We can check velocity change?
    # Initial speed 200. 
    # Friction is usually small.
    # If Thrust > 0, speed increases or stays high.
    # If Thrust == 0, speed decreases due to friction.
    # Friction in `gpu_physics.py` is (1-0.15) = 0.85 per step.
    # So if Thrust=0, Speed should drop to 200 * 0.85 = 170.
    
    env.step(actions, None)
    
    new_vel = torch.norm(env.physics.vel[0, 2]).item()
    print(f"Initial Speed: 200.0")
    print(f"New Speed: {new_vel:.1f}")
    
    expected_friction_speed = 200.0 * 0.85 
    print(f"Expected Speed if Thrust=0 (Friction 0.85): {expected_friction_speed:.1f}")
    
    if new_vel <= expected_friction_speed + 1.0:
        print(">> THRUST IS ZERO (Parking Confirmed)")
    else:
        print(">> THRUST IS ACTIVE (Failed to Park)")

    # 3. Scenario: Approaching but not close (Ramp check)
    # Dist to Gate ~ 1500. 
    # Ramp range 600..2500.
    # Factor ~ (1500-600)/1900 ~ 0.47.
    # Thrust should be ~50 * 0.47 ~ 23.
    # Speed will increase slightly or drop less than friction.
    
    print("\n[Scenario 2: Ramping]")
    start_pos_ramp = gate_pos_est - torch.tensor([1500.0, 0.0]) 
    env.physics.pos[:, 2] = start_pos_ramp
    env.physics.vel[:, 2] = torch.tensor([200.0, 0.0])
    
    # Ensure Runner is far enough to avoid Ramming (close_mask < 800)
    # Bot at CP-2000. Runner needs to be > 800 away. CP-3000.
    env.physics.pos[:, 0] = cp_pos - torch.tensor([3000.0, 0.0]) 
    
    env.step(actions, None)
    new_vel_ramp = torch.norm(env.physics.vel[0, 2]).item()
    print(f"Dist 1500. New Speed: {new_vel_ramp:.1f}")
    
    if new_vel_ramp > expected_friction_speed + 5.0:
         print(">> THRUST IS REDUCED BUT ACTIVE (Ramping Confirmed)")
    else:
         print(">> THRUST IS ZERO OR LOW")

if __name__ == "__main__":
    verify_bot_braking()
