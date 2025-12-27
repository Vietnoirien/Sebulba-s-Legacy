
import torch
import math
from simulation.env import PodRacerEnv, RW_ORIENTATION, RW_WRONG_WAY, DEFAULT_REWARD_WEIGHTS

def test_angle_logic():
    print("--- Verifying Angle Logic ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = PodRacerEnv(1, device=device, start_stage=0)
    
    # 1. Setup Scenario
    # Pod at (0, 0)
    # Checkpoint at (1000, 0) -> Target Angle 0 degrees.
    env.physics.pos[0, 0] = torch.tensor([0.0, 0.0], device=device)
    env.checkpoints[0, 1] = torch.tensor([1000.0, 0.0], device=device)
    env.next_cp_id[0, 0] = 1 # Target CP 1
    
    # Weights
    weights = torch.zeros((1, 10), device=device)
    weights[0, RW_ORIENTATION] = 1.0 # Simple unit weight
    weights[0, RW_WRONG_WAY] = 2.0   # 2x penalty
    
    # Helper to step and get orientation reward
    def get_reward_for_angle(angle_deg):
        # Set Angle
        env.physics.angle[0, 0] = angle_deg
        
        # We need to call step but isolate orientation reward logic.
        # step() computes it internally.
        # Let's inspect env.py logic or just run step with ONLY orientation weight set.
        # And zero velocity to avoid noise?
        env.physics.vel[0, 0] = 0.0
        env.prev_dist[0, 0] = 1000.0 # Same dist
        
        # Zero actions
        actions = torch.zeros((1, 4, 4), device=device)
        
        rewards, dones, infos = env.step(actions, reward_weights=weights)
        
        # Reward for Team 0 is accumulated.
        return rewards[0, 0].item()

    # Case 1: 0 degrees (Perfect)
    r_0 = get_reward_for_angle(0.0)
    print(f"Angle 0 deg (Aligned): {r_0:.4f} (Expected ~1.0)")
    
    # Case 2: 30 degrees (Cos=0.866)
    # Threshold=0.5. (0.866 - 0.5) / 0.5 = 0.732
    r_30 = get_reward_for_angle(30.0)
    print(f"Angle 30 deg: {r_30:.4f} (Expected ~0.732)")
    
    # Case 3: 60 degrees (Cos=0.5)
    # Threshold=0.5. (0.5 - 0.5) = 0.0
    r_60 = get_reward_for_angle(60.0)
    print(f"Angle 60 deg: {r_60:.4f} (Expected 0.0)")
    
    # Case 4: 90 degrees (Cos=0.0)
    # (0 - 0.5) < 0 -> Clamped 0 positive.
    # Neg mask? 0 < 0 False.
    r_90 = get_reward_for_angle(90.0)
    print(f"Angle 90 deg: {r_90:.4f} (Expected 0.0)")
    
    # Case 5: 180 degrees (Cos=-1.0)
    # Pos: 0.
    # Neg: -1.0 * Weight(2.0) = -2.0.
    r_180 = get_reward_for_angle(180.0)
    print(f"Angle 180 deg (Wrong Way): {r_180:.4f} (Expected -2.0)")
    
    if r_0 > 0.9 and r_60 < 0.01 and r_180 < -1.0:
        print("\n>>> VERIFICATION PASSED <<<")
    else:
        print("\n>>> VERIFICATION FAILED <<<")

if __name__ == "__main__":
    test_angle_logic()
