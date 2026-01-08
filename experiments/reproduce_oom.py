import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from training.ppo import PPOTrainer
from config import TrainingConfig

def test_buffer_churn():
    print("Initializing Trainer (Mocking)...")
    # minimal config
    config = TrainingConfig()
    config.pop_size = 2
    config.num_envs = 10 # small
    # derive envs_per_agent
    # config.envs_per_agent is a computed property
    config.device = 'cpu' 
    
    try:
        trainer = PPOTrainer(config=config, device='cpu')
    except Exception as e:
        print(f"Failed to init trainer: {e}")
        return

    print(f"Num Envs: {trainer.config.num_envs}")
    
    print("\n--- Call 1: Allocate Buffers ---")
    trainer.allocate_buffers()
    if not trainer.agent_batches:
        print("Error: agent_batches empty")
        return
        
    obj1 = trainer.agent_batches[0]['self_obs']
    id1 = id(obj1)
    print(f"Batch 0 Obj ID: {id1}")
    
    print("\n--- Call 2: Allocate Buffers (Same Config) ---")
    trainer.allocate_buffers()
    obj2 = trainer.agent_batches[0]['self_obs']
    id2 = id(obj2)
    print(f"Batch 0 Obj ID: {id2}")
    
    if id1 != id2:
        print("\n[RESULT] FAIL: Buffers were re-allocated despite same config!")
        print("This confirms the unnecessary memory churn.")
    else:
        print("\n[RESULT] SUCCESS: Buffers were reused.")

if __name__ == "__main__":
    test_buffer_churn()
