
import torch
import sys
import os

# Ensure we can import from current dir
sys.path.append(os.getcwd())

from simulation.env import PodRacerEnv

def test_reward():
    device = 'cpu' # Use CPU for debug
    env = PodRacerEnv(1, device=device)
    
    # 1. Reset
    env.reset()
    
    # 2. Setup Scenario
    # CP0 is start. CP1 is target.
    # Get CP1 pos
    cp1 = env.checkpoints[0, 1]
    
    print(f"CP1 Pos: {cp1}")
    
    # Place Pod 0 very close to CP1 (590 units away < 600 radius)
    # But wait, step happens first. Then Physics. Then Logic.
    # Logic checks dist AFTER physics.
    # So we place it such that physics moves it INTO the circle.
    
    # Place it 700 units away.
    # Move towards it at 200 units/step.
    # Next step: 500 units away -> PASSED.
    
    # Vector from CP1 to 0.0
    # Just place Pod at CP1_x - 700, CP1_y
    env.physics.pos[0, 0, 0] = cp1[0] - 800
    env.physics.pos[0, 0, 1] = cp1[1]
    
    env.physics.vel[0, 0, 0] = 300 # Moving towards CP
    env.physics.vel[0, 0, 1] = 0
    
    # Update prev_progress
    # We need prev_dist to reflect the distance BEFORE the step.
    # step() calculates new_dist (after physics) and compares to prev_dist.
    # prev_dist must be correct for current pos.
    env.update_progress_metric(torch.arange(1))
    
    # Verify initial prev_dist
    p_dist = env.prev_dist[0, 0].item()
    print(f"Initial Dist: {p_dist}")
    
    # 3. Step
    # Action: Thrust 100, Angle 0
    # [Thrust, Angle, Shield, Boost]
    # Thrust 1.0 -> 100.
    actions = torch.zeros((1, 4, 4))
    actions[0, 0, 0] = 1.0 # Full Thrust
    
    print("Stepping...")
    rewards, dones = env.step(actions)
    
    # 4. Check results
    new_pos = env.physics.pos[0, 0]
    new_dist_to_cp1 = torch.norm(cp1 - new_pos).item()
    print(f"New Pos: {new_pos}")
    print(f"Dist to CP1: {new_dist_to_cp1}") # Should be ~500 < 600
    
    # Role check
    print(f"Next CP ID: {env.next_cp_id[0, 0]}") # Should be 2
    
    r0 = rewards[0, 0].item()
    print(f"Reward Team 0: {r0}")
    
    # Expected Reward:
    # Dense: (PrevDist - NewDistToCP1) * Scale * (1-Tau)
    # CP Reward: 500
    
    expected_dense = (p_dist - new_dist_to_cp1) * 0.5
    print(f"Expected Dense (approx): {expected_dense}")
    print(f"Expected Total (approx): {expected_dense + 500}")
    
    if r0 > 400:
        print("SUCCESS: Positive Reward Received!")
    else:
        print("FAILURE: Low or Negative Reward.")

if __name__ == "__main__":
    test_reward()
