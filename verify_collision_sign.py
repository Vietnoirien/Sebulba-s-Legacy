
import torch
import math
from simulation.gpu_physics import GPUPhysics
from simulation.env import PodRacerEnv
from config import *

def test_collision_sign():
    device = "cpu"
    num_envs = 2
    
    # 1. Setup Env with Dummy Config
    env = PodRacerEnv(num_envs, device=device, start_stage=STAGE_DUEL_FUSED)
    
    # Enable Bots to ensure target logic is active (though we override positions)
    env.config.use_bots = True
    
    # Reset
    env.reset()
    
    # --- SCENARIO 1: HELPING (Rear-End) ---
    # Env 0
    # Blocker (Pod 1, assumes Team 0 is running blocker on pod 1? No, Env Logic varies)
    # Let's force roles.
    # Pod 0: Runner (Team 0). Pod 1: Blocker (Team 0).
    # Pod 2: Runner (Team 1). Pod 3: Blocker (Team 1).
    
    # We want Me (Blocker, Pod 1) to hit Enemy Runner (Pod 2).
    # Setup Positions:
    # CP at (10000, 0).
    # Enemy Runner (Pod 2) at (5000, 0). Vel (100, 0).
    # Me Blocker (Pod 1) at (4500, 0) (Behind). Vel (500, 0) (Faster, Chasing).
    
    # Set Roles
    env.is_runner.fill_(False)
    env.is_runner[0, 2] = True # Enemy is Runner
    env.is_runner[0, 1] = False # Me is Blocker
    
    # Set Next CP
    # Force Next CP to be far ahead
    env.checkpoints[0, 1, 0] = 10000.0
    env.checkpoints[0, 1, 1] = 0.0
    env.next_cp_id[0, 2] = 1 # Enemy aiming at CP 1
    
    # Set Physics
    env.physics.pos[0, 2] = torch.tensor([5000.0, 0.0]) # Enemy
    env.physics.vel[0, 2] = torch.tensor([100.0, 0.0])
    
    env.physics.pos[0, 1] = torch.tensor([4400.0, 0.0]) # Me (600u behind) - Will end at 5000 vs 5100
    
    # Move others away
    env.physics.pos[0, 0] = torch.tensor([-9999.0, 0.0])
    env.physics.pos[0, 3] = torch.tensor([-9999.0, 0.0])
    
    # --- SCENARIO 2: BLOCKING (Head-On) ---
    # Env 1
    # Me Blocker (Pod 1) at (6000, 0) (Ahead). Vel (-500, 0) (Ramming).
    # Enemy Runner (Pod 2) at (5000, 0). Vel (100, 0).
    
    env.is_runner[1, 2] = True
    env.is_runner[1, 1] = False
    
    env.checkpoints[1, 1, 0] = 10000.0
    env.checkpoints[1, 1, 1] = 0.0
    env.next_cp_id[1, 2] = 1
    
    env.physics.pos[1, 2] = torch.tensor([5000.0, 0.0]) # Enemy
    env.physics.vel[1, 2] = torch.tensor([100.0, 0.0])
    
    env.physics.pos[1, 1] = torch.tensor([5600.0, 0.0]) # Me (600u ahead) - Will end at 5200 vs 5100
    
    # Step
    # Actions: No thrust/turn, just coast into collision
    actions = torch.zeros((num_envs, 4, 4))
    
    print("\n--- PRE STEP ---")
    print(f"Env 0 (Helping): Me {env.physics.pos[0,1]} Vel {env.physics.vel[0,1]} -> Enemy {env.physics.pos[0,2]} Vel {env.physics.vel[0,2]} -> CP {env.checkpoints[0,1]}")
    print(f"Env 1 (Blocking): Me {env.physics.pos[1,1]} Vel {env.physics.vel[1,1]} -> Enemy {env.physics.pos[1,2]} Vel {env.physics.vel[1,2]} -> CP {env.checkpoints[1,1]}")
    
    # We need to hook into the reward calc loops to print debug info.
    # Or just inspect state after step, but rewards are transient.
    # I will rely on the returned 'rewards' tensor and verify magnitude/sign.
    
    # Set weights to ISOLATE collision reward
    # w_col_block = 1.0. Others 0.
    weights = torch.zeros((num_envs, 20))
    # RW_COLLISION_BLOCKER is 6
    weights[:, 6] = 1.0 
    
    rewards, dones, infos = env.step(actions, weights, tau=0.0)
    
    print("\n--- POST STEP ---")
    
    # Manual Physics Check (Did they bounce?)
    print(f"Env 0 (Helping): Me Vel {env.physics.vel[0,1]} | Enemy Vel {env.physics.vel[0,2]} (Should be boosted)")
    print(f"Env 1 (Blocking): Me Vel {env.physics.vel[1,1]} | Enemy Vel {env.physics.vel[1,2]} (Should be stopped/reversed)")
    
    # Reward Check
    # Me is Pod 1
    rew_help = rewards[0, 1].item()
    rew_block = rewards[1, 1].item()
    
    col_flags = infos["collision_flags"]
    print(f"Collision Flags: {col_flags}")

    print(f"\n--- REWARDS (Pod 1 - Blocker) ---")
    print(f"Helping Reward (Env 0): {rew_help:.4f}")
    print(f"Blocking Reward (Env 1): {rew_block:.4f}")
    
    if rew_help > 0:
        print("FAIL: Helping was REWARDED (Positive). Sign Error Confirmed.")
    else:
        print("SUCCESS?: Helping was PENALIZED (Negative). User might be right?")
        
    if rew_block > 0:
        print("SUCCESS: Blocking was REWARDED (Positive).")
    else:
        print("FAIL: Blocking was PENALIZED (Negative).")

test_collision_sign()
