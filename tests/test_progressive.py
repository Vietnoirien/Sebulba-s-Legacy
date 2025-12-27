
import torch
from simulation.env import PodRacerEnv, STAGE_SOLO

def test_progressive_rewards():
    print("Initializing Env...")
    env = PodRacerEnv(num_envs=4, device='cpu', start_stage=STAGE_SOLO)
    print("Resetting...")
    env.reset()
    
    # We need to simulate passing a CP.
    # The logic is inside step(), triggered by distance check.
    # We can fake the positions to force a pass.
    
    print("Forcing Checkpoint Pass 1 (CP 1)...")
    # All pods start at CP 0. Next is CP 1.
    # Get CP 1 pos
    target_pos = env.checkpoints[0, 1] 
    # Move Pod 0 to Target
    env.physics.pos[0, 0] = target_pos
    
    # Step to trigger reward
    actions = torch.zeros((4, 4, 4))
    rewards, _ = env.step(actions)
    
    # Buffer should have received reward.
    # Streak should be 1. Reward = 500 + 0 = 500.
    buf1 = env.cp_reward_buffer[0, 0].item()
    passed1 = env.cps_passed[0, 0].item()
    print(f"Pass 1: Buffer added approx {buf1} (Expected ~500), Passed Count: {passed1}")
    
    # Manually drain buffer to clear it for next test (simulating passage of time)
    env.cp_reward_buffer[0, 0] = 0.0
    
    print("Forcing Checkpoint Pass 2 (CP 2)...")
    # Next is CP 2
    target_pos_2 = env.checkpoints[0, 2]
    env.physics.pos[0, 0] = target_pos_2
    
    rewards, _ = env.step(actions)
    
    # Streak should be 2. Reward = 500 + 1*500 = 1000.
    buf2 = env.cp_reward_buffer[0, 0].item()
    passed2 = env.cps_passed[0, 0].item()
    print(f"Pass 2: Buffer added approx {buf2} (Expected ~1000), Passed Count: {passed2}")
    
    if passed2 == 2 and buf2 > buf1:
        print("Progressive Reward Verified.")
    else:
        print("Verification Failed.")

if __name__ == "__main__":
    try:
        test_progressive_rewards()
        print("Test Success")
    except Exception as e:
        print("Test Failed")
        print(e)
