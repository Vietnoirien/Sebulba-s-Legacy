import torch
import time
from simulation.env import PodRacerEnv, STAGE_SOLO

def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    start_init = time.time()
    # Create env but don't reset yet to measure pure reset time if possible, 
    # but init calls reset.
    env = PodRacerEnv(num_envs=1000, device=device, start_stage=STAGE_SOLO)
    end_init = time.time()
    print(f"Init (First Reset) Time: {end_init - start_init:.4f}s")
    
    # Measure repeated resets
    start = time.time()
    for i in range(10):
        env.reset()
        torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 10.0
    print(f"Average Reset Time (1000 envs): {avg_time:.4f}s")
    
    # Check failure rate (fallback usage)
    # If fallback used, points are in a line.
    # Check if checkpoints match the line pattern
    # pattern: px = 2000 + i*3000, py = HEIGHT/2
    
    cps = env.checkpoints
    # Check first environment for pattern
    # (Just rough check)
    
if __name__ == "__main__":
    benchmark()
