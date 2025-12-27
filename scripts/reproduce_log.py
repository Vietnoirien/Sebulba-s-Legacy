
import sys
import os
import time
from unittest.mock import MagicMock

# Mock necessary modules
sys.modules['simulation.env'] = MagicMock()
sys.modules['models.deepsets'] = MagicMock()
sys.modules['training.self_play'] = MagicMock()
sys.modules['training.normalization'] = MagicMock()
sys.modules['config'] = MagicMock()

# Define constants needed by ppo.py imports
import training.ppo as ppo_module

# Patch constants to run fast
ppo_module.NUM_ENVS = 2
ppo_module.POP_SIZE = 2
ppo_module.ENVS_PER_AGENT = 1
ppo_module.NUM_STEPS = 10
ppo_module.TOTAL_TIMESTEPS = 100
ppo_module.DEFAULT_REWARD_WEIGHTS = {}

from training.ppo import PPOTrainer

def mock_callback(msg):
    print(f"CALLBACK: {msg}")

def test_logging():
    print("Initializing Trainer...")
    trainer = PPOTrainer(device='cpu', logger_callback=mock_callback)
    
    # Mock environment to return dummy data
    trainer.env = MagicMock()
    trainer.env.get_obs.return_value = (
        MagicMock(), MagicMock(), MagicMock() # self, ent, cp
    )
    trainer.env.step.return_value = (
        MagicMock(), MagicMock(), MagicMock() # rewards, dones, infos
    )
    # Mock tensors returned by env to avoid shape errors
    import torch
    trainer.env.curriculum_stage = 0
    
    # Patch internals to avoid complex attribute errors
    trainer.rms_self = MagicMock()
    trainer.rms_ent = MagicMock()
    trainer.rms_cp = MagicMock()
    trainer.rms_ret = MagicMock()
    
    for p in trainer.population:
        p['agent'] = MagicMock()
        p['agent'].get_action_and_value.return_value = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        p['agent'].get_value.return_value = torch.zeros(1)
        p['optimizer'] = MagicMock()

    print("Starting Train Loop...")
    try:
        # We need to run it for a bit.
        # But wait, env.step needs to return tensors of correct shape?
        # This might be hard to mock perfectly.
        # Let's just mock log() method directly to see when it's called.
        
        orig_log = trainer.log
        trainer.log = MagicMock(side_effect=orig_log)
        
        # We can't easily run train_loop without a working env.
        # But we can inspect the code structure.
        pass
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test_logging()
