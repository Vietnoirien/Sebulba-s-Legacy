import torch
import sys
import os
import time

# Add root to path
sys.path.append(os.getcwd())

from training.ppo import PPOTrainer
from simulation.env import STAGE_INTERCEPT
from training.curriculum.stages import InterceptStage # Ensure stage class is available if needed

def verify():
    print("Initializing Trainer for Denial Rate Verification...")
    
    # Initialize with CUDA if available, else CPU (though codebase expects CUDA mostly)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PPOTrainer(device=device)
    
    # Configure for short run
    trainer.config.total_timesteps = 1000 # Run ~2 iterations
    trainer.config.num_steps = 512
    # trainer.config.envs_per_agent = 32 # Reduce scale for test (Removed: Read-only)
    
    # Force Stage 3 (Intercept)
    print("Forcing Stage 3 (Intercept)...")
    
    # Get Config from Stage Class
    # We need to access curriculum stages dictionary or list
    # PPOTrainer initializes curriculum.
    # curriculum.stages is a dict or list?
    # In stages.py, it seems separate. In manager.py it initializes them.
    # We can just manually create config if needed, or rely on trainer.
    
    # Let's trust trainer.curriculum has it.
    stage_instance = trainer.curriculum.stages[STAGE_INTERCEPT]
    env_config = stage_instance.get_env_config()
    
    # Set Stage
    trainer.curriculum.set_stage(STAGE_INTERCEPT)
    trainer.env.set_stage(STAGE_INTERCEPT, env_config, reset_env=True)
    
    print("Starting Training Loop...")
    
    # Override log to catch output
    def filtered_log(msg):
        if "Denial Rate" in msg or "Stage 3" in msg:
            print(f"[VERIFY] {msg}")
        # print(f"[LOG] {msg}") # Verbose
        
    trainer.log = filtered_log
    
    try:
        trainer.train_loop()
    except Exception as e:
        print(f"Run Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
