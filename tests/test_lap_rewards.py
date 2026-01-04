
import torch
import sys
import os

# Create a clean test env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.env import PodRacerEnv
from config import LAP_REWARD_MULTIPLIER, RW_PROGRESS

def test_rewards():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    env = PodRacerEnv(num_envs=1, device=device)
    env.reset()
    
    # Disable Progress Reward to avoid Teleport noise
    # RW_PROGRESS is index 4.
    # Env uses passed weights or default.
    # We can pass weights with 0 for progress.
    # Construct weights
    from config import DEFAULT_REWARD_WEIGHTS
    weights = DEFAULT_REWARD_WEIGHTS.copy()
    weights[RW_PROGRESS] = 0.0 # Disable
    
    # Convert to tensor [1, 15]
    w_tensor = torch.zeros((1, 15), device=device)
    for k, v in weights.items():
        w_tensor[0, k] = v
        
    print("\n--- Test 1: Normal Checkpoint Reward ---")
    env_idx = 0
    pod_idx = 0
    
    target_id = 1
    env.next_cp_id[env_idx, pod_idx] = target_id
    target_pos = env.checkpoints[env_idx, target_id]
    
    # Teleport
    env.physics.pos[env_idx, pod_idx] = target_pos
    
    # Step
    rewards, dones, infos = env.step(torch.zeros((1, 4, 4), device=device), w_tensor)
    
    r = rewards[env_idx, pod_idx].item()
    passed = infos["checkpoints_passed"][env_idx, pod_idx].item()
    print(f"Passed Info: {passed}")
    print(f"Reward Received: {r}")
    
    if 450.0 < r < 550.0:
        print("PASS: Checkpoint Reward correct.")
    else:
        print(f"FAIL: Expected ~500, got {r}")

    print("\n--- Test 2: Lap 1 Reward ---")
    # Reset reward state? Env accumulates? No, step returns step reward.
    
    env.next_cp_id[env_idx, pod_idx] = 0
    env.laps[env_idx, pod_idx] = 0
    target_pos = env.checkpoints[env_idx, 0]
    
    env.physics.pos[env_idx, pod_idx] = target_pos
    
    rewards, dones, infos = env.step(torch.zeros((1, 4, 4), device=device), w_tensor)
    
    r1 = rewards[env_idx, pod_idx].item()
    passed1 = infos["checkpoints_passed"][env_idx, pod_idx].item()
    laps1 = infos["laps_completed"][env_idx, pod_idx].item()
    
    print(f"Passed: {passed1}, Laps Completed: {laps1}")
    print(f"Reward: {r1}")
    
    # Exp: 2000
    if 1800.0 < r1 < 2200.0:
         print("PASS: Lap 1 Reward correct.")
    else:
         print(f"FAIL: Expected ~2000, got {r1}")

    print("\n--- Test 3: Lap 2 Reward ---")
    # Force state to Lap 1 (Complete 1, working on 2? No, finishing Lap 2 means entering Lap 2?)
    # "Lap 1 Completion" means Laps goes 0 -> 1.
    # "Lap 2 Completion" means Laps goes 1 -> 2.
    
    env.laps[env_idx, pod_idx] = 1 # Finished 1 lap already.
    env.next_cp_id[env_idx, pod_idx] = 0
    
    # Move slightly away and back?
    # Or strict teleport again (might fail if "already passed" logic exists? No, cp logic is stateless per step mostly)
    
    env.physics.pos[env_idx, pod_idx] = target_pos
    
    rewards, dones, infos = env.step(torch.zeros((1, 4, 4), device=device), w_tensor)
    
    r2 = rewards[env_idx, pod_idx].item()
    print(f"Reward: {r2}")
    
    # Exp: 3000 (2000 * 1.5)
    if 2800.0 < r2 < 3200.0:
         print("PASS: Lap 2 Reward correct.")
    else:
         print(f"FAIL: Expected ~3000, got {r2}")

if __name__ == "__main__":
    test_rewards()
