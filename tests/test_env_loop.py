
import time
import torch
from simulation.env import PodRacerEnv, STAGE_SOLO

def test_loop():
    print("Initializing Env...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create Env
    env = PodRacerEnv(num_envs=4096, device=device, start_stage=STAGE_SOLO)
    print("Resetting...")
    start_reset = time.time()
    env.reset()
    print(f"Reset took {time.time() - start_reset:.4f}s")
    
    print("Starting Step Loop (500 steps)...")
    actions = torch.zeros((4096, 4, 4), device=device)
    
    start_loop = time.time()
    for i in range(500):
        if i % 100 == 0:
            print(f"Step {i}")
        rewards, dones, infos = env.step(actions)
        
        if dones.any():
            env.reset(torch.nonzero(dones).squeeze(-1))
            
    total_time = time.time() - start_loop
    sps = (500 * 128) / total_time
    print(f"Loop finished in {total_time:.2f}s")
    print(f"Approx SPS (Physics only): {sps:.2f}")

if __name__ == "__main__":
    test_loop()
