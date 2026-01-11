
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from simulation.env import PodRacerEnv, STAGE_DUEL_FUSED, RW_ORIENTATION, RW_ZONE, RW_COLLISION_BLOCKER, RW_DENIAL
from config import EnvConfig

def test_orientation_vs_intercept():
    device = "cpu"
    env = PodRacerEnv(num_envs=1, device=device, start_stage=STAGE_DUEL_FUSED)
    
    # Setup: Blocker closing in on Intercept Point
    env.reset()
    env.is_runner[:] = False
    env.is_runner[:, 2] = True
    
    # Positions
    # Me at (0, 0), moving (500, 0)
    # Target Intercept at (2000, 0)
    # Enemy at (2000, 0), moving away? let's make it static for simple intercept calc.
    # If Enemy static, Intercept = Enemy Pos.
    
    env.physics.pos[:, 0] = torch.tensor([[0.0, 0.0]])
    env.physics.pos[:, 2] = torch.tensor([[2000.0, 0.0]])
    env.physics.vel[:, 0] = torch.tensor([[500.0, 0.0]]) # Closing speed 500
    env.physics.vel[:, 2] = torch.tensor([[0.0, 0.0]])
    env.physics.angle[:, 0] = 0.0 # Facing Target (Perfect Orientation)
    
    # Set weights
    weights = torch.zeros((1, 25))
    weights[:, RW_ORIENTATION] = 1.0
    
    # We need to manually inspect the separate components.
    # Since env.step sums them into rewards_indiv, we can't easily see them unless we zero others?
    # But Intercept Reward is added to rewards_indiv directly, not via weights index? 
    # Let's check env.py line 1600.
    # rewards_indiv[:, i] += intercept_rew * is_block.float()
    # It adds directly! It doesn't use a weight from `reward_weights`?
    # Wait, line 1598: `intercept_rew = delta * valid_delta * INTERCEPT_SCALE * dense_mult`
    # INTERCEPT_SCALE is from config.
    
    # So if we set all weights to 0, we should see ONLY Intercept Reward (and maybe penalties).
    # RW_ORIENTATION is added via: `rewards_indiv[:, i] += pos_score * w_orient * dense_mult`
    
    # Test 1: Orientation Only (Mock Intercept as 0 via config?)
    print("\n--- TEST: Separating Rewards ---")
    
    # Step 1: Run with valid movement (Init Prev Dist)
    rewards, dones, info = env.step(torch.zeros((1, 4, 4)), weights, tau=0.0)
    print(f"Step 1 Reward (Init): {rewards[0, 0].item()}")
    
    # Step 2: Continue movement
    # Physics is stateful, so we just step again.
    # Pos was updated in Step 1 (0 -> 500 if tau=0? Wait, step calls physics_step)
    # Env step length? 1.0s? 
    # Let's assume positions updated.
    
    rewards, dones, info = env.step(torch.zeros((1, 4, 4)), weights, tau=0.0)
    total_rew = rewards[0, 0].item()
    print(f"Step 2 Reward (Weights=0 except Orient=1): {total_rew}")
    
    # Calculate Orientation Component
    # Angle 0. Target (2000, 0) -> Angle 0. Diff=0. Cos(0)=1.
    # Pos Score = (1 - 0.5) / 0.5 = 1.0.
    # Orient Rew = 1.0 * 1.0 * 1.0 = 1.0.
    
    # Verify Intercept Component
    # Dist Old: 2000.
    # Dist New: 1500.
    # Delta: 500.
    # Scale: 1.0.
    # Intercept Rew: 500.0.
    
    # Expected Total: 501.0.
    
    print(f"Estimated Orientation: 1.0")
    print(f"Estimated Intercept: 500.0")
    
if __name__ == "__main__":
    test_orientation_vs_intercept()
