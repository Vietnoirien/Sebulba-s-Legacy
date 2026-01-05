import torch
import sys
import os
import time

# Add root to path
sys.path.append(os.getcwd())

from training.ppo import PPOTrainer
from config import TrainingConfig, STAGE_TEAM

def verify_fix():
    print("Initializing Trainer...")
    trainer = PPOTrainer(device='cuda')
    
    # 1. Load Gen 66
    gen_path = "data/generations/gen_66"
    print(f"Loading Generation 66 from {gen_path}...")
    
    loaded_count = 0
    for p in trainer.population:
        agent_id = p['id']
        path = os.path.join(gen_path, f"agent_{agent_id}.pt")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=trainer.device)
            p['agent'].load_state_dict(state_dict)
            loaded_count += 1
            
    print(f"Loaded {loaded_count}/{len(trainer.population)} agents.")
    
    # Sync to Vectorized
    trainer.sync_agents_to_vectorized()
    
    # 2. Force Stage 3
    print("Forcing Stage 3 (Team)...")
    trainer.curriculum.set_stage(STAGE_TEAM)
    trainer.env.set_stage(STAGE_TEAM, trainer.curriculum.current_stage.get_env_config(), reset_env=True)
    
    # 3. Mitosis (Simulate what happens on transition)
    # We assume Gen 66 might already have Mitosis applied if it was "freshly arriving"?
    # User said "freshly arriving in stage 3 AFTER mitosis".
    # So weights should be good. 
    # But we should ensure Optimizer state is reset if we were continuing.
    # Since we created a NEW trainer, optimizer is fresh.
    
    # 4. Run Training Loop for a few steps
    print("Starting Training Loop (1 Iteration)...")
    trainer.config.update_epochs = 2 # Reduce for speed
    
    # Manually run one iteration logic
    trainer.allocate_buffers()
    trainer.env.reset()
    
    # Initialize Obs
    obs_data = trainer.env.get_obs()
    all_self, all_tm, all_en, all_cp = obs_data
    
    # Run Steps
    for step in range(100): # Run 100 steps
        # ... logic consistent with ppo.py inner loop ...
        # Actually better to just call train_loop with a stop condition?
        # But train_loop has a while loop.
        pass

    # Run
    try:
        trainer.config.num_steps = 128
        trainer.train_loop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Run Error: {e}")


if __name__ == "__main__":
    verify_fix()
