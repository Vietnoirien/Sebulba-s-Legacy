import torch
from simulation.env import PodRacerEnv

def test_orientation_reward():
    env = PodRacerEnv(num_envs=1, device='cuda', start_stage=0)
    
    print("\n--- Testing Orientation Reward ---")
    
    # Reset
    env.reset()
    
    # Pod 0 is active in solo
    # Get its position and next checkpoint
    pos = env.physics.pos[0, 0]
    next_cp_idx = env.next_cp_id[0, 0]
    target = env.checkpoints[0, next_cp_idx]
    
    print(f"Pos: {pos.tolist()}")
    print(f"Target: {target.tolist()}")
    
    # Calculate vector
    diff = target - pos
    target_angle_rad = torch.atan2(diff[1], diff[0])
    target_angle_deg = torch.rad2deg(target_angle_rad)
    print(f"Target Angle: {target_angle_deg.item():.2f}")
    
    # Case 1: Perfect Alignment
    print("\nCase 1: Aligning Pod to Target...")
    env.physics.angle[0, 0] = target_angle_deg
    
    # Step with 0 actions (just to trigger reward calc)
    actions = torch.zeros((1, 4, 4), device='cuda')
    # Use custom reward config to isolate orientation
    reward_config = {
        "tau": 0.0,
        "beta": 0.0,
        "weights": {
            "orientation": 10.0, # High weight to see it
            "velocity": 0.0,
            "step_penalty": 0.0
        }
    }
    
    rewards, _ = env.step(actions, reward_config=reward_config)
    
    # Expected reward: 1.0 * 10.0 = 10.0
    print(f"Reward (Perfect Align): {rewards[0, 0].item():.4f} (Expected ~10.0)")
    
    # Case 2: 90 Degrees Off
    print("\nCase 2: 90 Degrees Off...")
    env.physics.angle[0, 0] = target_angle_deg + 90.0
    rewards, _ = env.step(actions, reward_config=reward_config)
    print(f"Reward (90 Deg): {rewards[0, 0].item():.4f} (Expected ~0.0)")
    
    # Case 3: 180 Degrees Off
    print("\nCase 3: 180 Degrees Off...")
    env.physics.angle[0, 0] = target_angle_deg + 180.0
    rewards, _ = env.step(actions, reward_config=reward_config)
    print(f"Reward (180 Deg): {rewards[0, 0].item():.4f} (Expected ~-10.0)")
    
    print("--- Test Done ---")

if __name__ == "__main__":
    test_orientation_reward()
