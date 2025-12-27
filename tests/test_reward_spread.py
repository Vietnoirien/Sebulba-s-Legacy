
import torch
from simulation.env import PodRacerEnv, STAGE_SOLO

def test_reward_spreading():
    print("Initializing Env...")
    env = PodRacerEnv(num_envs=4, device='cpu', start_stage=STAGE_SOLO)
    print("Resetting...")
    env.reset()
    
    # Manually Inject Reward into Buffer to simulate checkpoint pass
    # Since we can't easily force a physics pass without precise setup,
    # we just modify the buffer directly to test the payout logic.
    print("Injecting Reward into Buffer for Pod 0...")
    env.cp_reward_buffer[:, 0] = 2000.0 # Simulating 4 hits of 500 or just a big hit
    
    print("Stepping for 15 steps...")
    actions = torch.zeros((4, 4, 4)) 
    
    for s in range(15):
        rewards, done = env.step(actions)
        # Pod 0 is Team 0 (index 0)
        # We expect reward approx 2000/10 = 200 per step.
        # Plus dense rewards (small) + step penalty (0).
        
        r0 = rewards[:, 0].mean().item()
        buf0 = env.cp_reward_buffer[:, 0].mean().item()
        print(f"Step {s+1}: Reward Team 0 (Mean): {r0:.4f}, Buffer Rem: {buf0:.4f}")

if __name__ == "__main__":
    try:
        test_reward_spreading()
        print("Test Success")
    except Exception as e:
        print("Test Failed")
        print(e)
