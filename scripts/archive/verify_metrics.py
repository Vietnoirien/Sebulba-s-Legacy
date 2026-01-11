
import sys
import os
import torch
import time
import threading

# Add project root to path
sys.path.append(os.getcwd())

from training.ppo import PPOTrainer
from config import TrainingConfig, EnvConfig
from simulation.env import STAGE_DUEL_FUSED

def verify_metrics():
    print("--- Verifying Unified Metrics ---")
    
    # 1. Setup Config
    # Short run: 1 iteration, 128 steps
    train_config = TrainingConfig()
    train_config.total_timesteps = 2048
    train_config.num_steps = 512 # Standard
    train_config.pop_size = 32
    train_config.num_envs = 64
    
    env_config = EnvConfig()
    
    # 2. Initialize Trainer
    trainer = PPOTrainer(train_config, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Force Stage to Duel Fused to see Blocker Score
    trainer.env.curriculum_stage = STAGE_DUEL_FUSED
    trainer.curriculum.current_stage_idx = 2 # Approx
    
    # 4. Run Training
    print("Starting Training Loop (Short)...")
    try:
        # Run in main thread because signal handling might be tricky in threads, 
        # but here we just want it to finish 1 iteration.
        trainer.train_loop()
    except Exception as e:
        print(f"Training crashed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n--- Verification Finished ---")
    print("Check output above for:")
    print("1. 'Blocker Sc' column in the table.")
    print("2. Values in 'Blocker Sc' (should be > 0 if collisions happen).")
    print("3. No crashes.")

if __name__ == "__main__":
    verify_metrics()
