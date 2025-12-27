
import sys
import os
import torch
import time

# Ensure we can import the project
sys.path.append(os.getcwd())

from simulation.env import PodRacerEnv, STAGE_DUEL
from training.ppo import PPOTrainer

def test_stage_1_config():
    print("Testing Stage 1 Configuration...")
    
    # Init Trainer
    trainer = PPOTrainer(device='cuda')
    
    # Force Stage 1
    trainer.env.curriculum_stage = STAGE_DUEL
    trainer.env.bot_difficulty = 0.0
    print(f"Force Stage: {trainer.env.curriculum_stage}, Difficulty: {trainer.env.bot_difficulty}")
    
    # Run a few steps
    print("Running training loop steps...")
    
    # We can't easily call train_loop because it's infinite.
    # We'll simulate the logic inside train_loop roughly or just run it for a second.
    # Actually, let's just inspect the env.step logic by calling it directly?
    
    # Mock Actions
    actions = torch.zeros((trainer.env.num_envs, 4, 4), device='cuda')
    actions[..., 0] = 100.0 # Full thrust
    
    # Step
    trainer.env.step(actions)
    print("Step 1 complete.")
    
    # Check if metrics are accumulating
    print("Metrics:", trainer.env.stage_metrics)
    
    # Simulate a "Win" to check curriculum logic
    # Manually increment recent wins
    trainer.env.stage_metrics["recent_wins"] = 4000
    trainer.env.stage_metrics["recent_games"] = 5001 # Trigger threshold > 5000
    
    print("Simulating metrics for Curriculum Check...")
    trainer.check_curriculum()
    
    # Logic: WR = 4000/5001 ~= 0.8. Thresh is 0.7.
    # Should increase difficulty.
    print(f"New Difficulty (Should be 0.05): {trainer.env.bot_difficulty}")
    
    if trainer.env.bot_difficulty > 0.0:
        print("SUCCESS: Difficulty increased.")
    else:
        print("FAILURE: Difficulty did not increase.")
        sys.exit(1)

if __name__ == "__main__":
    test_stage_1_config()
