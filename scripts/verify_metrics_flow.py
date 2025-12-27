
import torch
from simulation.env import PodRacerEnv, STAGE_SOLO
import numpy as np

def verify_metrics():
    print("Initializing Env...")
    env = PodRacerEnv(num_envs=32, device='cpu', start_stage=STAGE_SOLO)
    
    # 1. Check Initial State
    print(f"Initial steps_last_cp: {env.steps_last_cp.float().mean().item()}")
    
    # 2. Step 10 times (No actions)
    print("Stepping 10 times...")
    actions = torch.zeros((32, 4, 4))
    for _ in range(10):
        env.step(actions)
        
    # 3. Check if steps_last_cp increased
    mean_steps = env.steps_last_cp.float().mean().item()
    print(f"Steps after 10 steps (Mean): {mean_steps}")
    
    if mean_steps == 0:
        print("FAIL: steps_last_cp did not increment!")
    else:
        print(f"SUCCESS: steps_last_cp incremented to {mean_steps}")

    # 4. Force a checkpoint pass to verify reset
    print("Forcing Checkpoint Pass on Env 0 Pod 0...")
    # Teleport to next CP
    target_id = env.next_cp_id[0, 0]
    target_pos = env.checkpoints[0, target_id]
    env.physics.pos[0, 0] = target_pos # Teleport to center
    
    # Step to trigger detection
    _, _, infos = env.step(actions)
    
    # Check info
    passed = infos['checkpoints_passed'][0, 0].item()
    steps_taken = infos['cp_steps'][0, 0].item()
    print(f"Passed: {passed}, Steps Taken recorded in info: {steps_taken}")
    
    # Check if reset
    current_val = env.steps_last_cp[0, 0].item()
    print(f"Current steps_last_cp after pass: {current_val} (Should be 0)")
    
    if passed and steps_taken > 0 and current_val == 0:
        print("SUCCESS: Metric flow confirmed.")
    else:
        print("FAIL: Metric flow broken.")

if __name__ == "__main__":
    verify_metrics()
