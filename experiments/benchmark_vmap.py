
import time
import torch
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from training.ppo import PPOTrainer
from config import *

def benchmark():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    print("Initializing PPOTrainer (Vectorized)...")
    trainer = PPOTrainer()
    
    # Configure for benchmark
    # We use standard sizing: 128 agents * 32 envs = 4096 envs
    # Rollout length: 256 (standard)
    # This matches the full load.
    
    print(f"Pop Size: {trainer.config.pop_size}")
    print(f"Envs Per Agent: {trainer.config.envs_per_agent}")
    print(f"Num Steps: {trainer.config.num_steps}")
    
    print("Pre-Run warmup (1 Iteration)...")
    # Run 1 iteration to JIT compile/warmup
    try:
        trainer.train_loop(stop_event=None, telemetry_callback=None)
    except Exception as e:
        print(f"Warmup Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Starting Benchmark (3 Iterations)...")
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    for i in range(3):
        iter_start = time.time()
        trainer.train_loop(stop_event=None, telemetry_callback=None)
        iter_dur = time.time() - iter_start
        print(f"Iter {i+1}: {iter_dur:.2f}s")
        
    end_time = time.time()
    
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Max VRAM: {vram:.2f} GB")
    print(f"Total Time: {end_time - start_time:.2f}s")
    print(f"Avg Iter Time: {(end_time - start_time)/3:.2f}s")
    
    sps = (trainer.config.num_envs * trainer.config.num_steps * 3) / (end_time - start_time)
    print(f"Estimated SPS: {sps:.2f}")

if __name__ == "__main__":
    benchmark()
