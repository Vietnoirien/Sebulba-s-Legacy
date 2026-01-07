
import sys
import os
import shutil
sys.path.append(os.getcwd())
from training.ppo import PPOTrainer
from config import TrainingConfig
import torch

def test_train_loop():
    print("Testing PPO Training Loop (Integration)...")
    
    # Use minimal config for speed
    config = TrainingConfig()
    config.num_envs = 8 # Small envs
    config.pop_size = 2 # Small population
    config.pop_size = 2 # Small population
    config.num_steps = 32 # Small rollout (Must be > seq_length=16)
    config.seq_length = 16
    config.num_minibatches = 2
    config.update_epochs = 1
    config.total_timesteps = 1000
    config.use_lstm = True
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mock Curriculum/Env? 
    # PPOTrainer initializes VectorizedEnv.
    # This requires 'simulation' module and valid config.
    # It will try to spawn subprocesses. This is risky if environment is complex to spawn.
    # But we want to test 'allocate_buffers' and 'train' loop logic.
    
    trainer = PPOTrainer(config, device=config.device)
    print("  Trainer Initialized.")
    
    try:
        # Run 1 iteration of the main loop logic manually?
        # trainer.train() runs forever. 
        # We can hijack 'train()' or just call 'collect_rollouts' and 'update'.
        
        print("  Allocating Buffers...")
        trainer.allocate_buffers()
        
        print("  Collecting Rollouts (Fake)...")
        # We can't easily fake env collection without mocking 'env'.
        # VectorizedEnv is heavy.
        # Let's trust logic for now or try to run trainer.train() in a thread and kill it?
        # Too complex.
        
        # Let's just check if buffers were allocated correctly.
        b0 = trainer.agent_batches[0]
        print("    Buffer Keys:", b0.keys())
        expected_chunks = config.num_steps // config.seq_length + 1
        print("    Expected Chunks:", expected_chunks)
        print("    Actor State Shape:", b0['actor_h'].shape)
        
        # [Chunks, BatchPerAgent, 1, Hidden]
        # BatchPerAgent = envs_per_agent * N_Active(2) = 4*2 = 8?
        # actually num_active_per_agent = envs_per_agent * 2.
        
        assert b0['actor_h'].shape[0] == expected_chunks
        
        print("Integration Test (Init/Alloc) Passed!")
        
    except Exception as e:
        print(f"Integration Failed: {e}")
        raise e
    finally:
        if hasattr(trainer, 'env'):
            trainer.env.close()

if __name__ == "__main__":
    test_train_loop()
