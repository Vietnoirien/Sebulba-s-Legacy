
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from simulation.env import PodRacerEnv
from models.deepsets import PodAgent
from config import TrainingConfig

def verify_map_embedding():
    print("Verifying Map Embedding Integration...")
    
    # 1. Initialize Config & Env
    config = TrainingConfig()
    env = PodRacerEnv(config.num_envs, device=config.device)
    env.reset()
    obs = env.get_obs()
    
    # 2. Check get_obs output
    # Expected: self, tm, en, cp, map
    if len(obs) != 5:
        print(f"FAILED: Expected 5 observation tensors, got {len(obs)}")
        return
        
    all_self, all_tm, all_en, all_cp, all_map = obs
    
    print(f"Map Obs Shape: {all_map.shape}")
    # Expected: [4, NumEnvs, MaxCP, 2]
    expected_shape = (4, config.num_envs, 6, 2)
    
    if all_map.shape != expected_shape:
        print(f"FAILED: Map Obs shape mismatch. Expected {expected_shape}, got {all_map.shape}")
        return
    else:
        print("SUCCESS: Map Obs shape correct.")
        
    # 3. Check Map Content (Basic)
    # Check if we have non-zero values (relative coords)
    if torch.all(all_map == 0):
        print("WARNING: Map tensor is all zeros. Might be valid if all CPs are at (0,0) relative (impossible).")
    else:
        print("SUCCESS: Map tensor contains data.")
        
    # 4. Initialize Agent
    print("Initializing Agent...")
    agent = PodAgent().to(config.device)
    
    # 5. Run Forward Pass
    print("Running Forward Pass...")
    
    # Slice for 1 agent (Agent 0)
    # Obs Input: [Batch, D]
    # Input to agent.get_action_and_value needs flattened batch
    
    # We simulate a "collection" step for Agent 0
    # Agent 0 sees:
    # Self: all_self[0]
    # Teammate: all_tm[0]
    # Enemy: all_en[0]
    # CP: all_cp[0]
    # Map: all_map[0]
    
    batch_size = config.num_envs
    
    s = all_self[0].to(config.device)
    t = all_tm[0].to(config.device)
    e = all_en[0].to(config.device)
    c = all_cp[0].to(config.device)
    m = all_map[0].to(config.device)
    
    try:
        # Check signature of get_action_and_value
        # It calls forward(self, tm, en, cp, map, ...)
        # We need to see if PodAgent.get_action_and_value was updated? 
        # I only updated PodActor.forward. 
        # Does PodAgent.get_action_and_value wrap it automatically?
        # Let's check PodAgent again.
        
        # NOTE: I might have missed updating PodAgent.get_action_and_value signature!
        # If so, this will fail. That's good verification.
        
        # Attempt call
        res = agent.get_action_and_value(s, t, e, c, m)
        print("SUCCESS: Agent Forward Pass successful.")
        
    except TypeError as e:
        print(f"FAILED: Forward Pass TypeError: {e}")
        print("Likely missing argument in PodAgent.get_action_and_value or PodActor.forward signature mismatch.")
    except Exception as e:
        print(f"FAILED: Forward Pass Error: {e}")
        import traceback
        traceback.print_exc()


from training.ppo import PPOTrainer

def verify_ppo_loop():
    print("\nVerifying PPO Loop...")
    config = TrainingConfig()
    # Reduced config for speed
    config.num_envs = 64
    config.num_steps = 64
    config.pop_size = 4
    config.num_minibatches = 4
    config.update_epochs = 1
    
    trainer = PPOTrainer(config, device=config.device)
    print("Trainer initialized.")
    
    # Run 1 iteration
    config.total_timesteps = config.num_envs * config.num_steps
    print("Starting 1 iteration...")
    trainer.train_loop()
    print("SUCCESS: PPO Loop finished.")

if __name__ == "__main__":
    verify_map_embedding()
    verify_ppo_loop()
