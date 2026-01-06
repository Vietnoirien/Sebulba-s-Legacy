import os
import shutil
import time
import torch
from unittest.mock import MagicMock
from training.ppo import PPOTrainer
from config import TrainingConfig, EnvConfig, CurriculumConfig

# Mock Classes to avoid loading full environment
class MockEnv:
    def __init__(self, stage=0):
        self.curriculum_stage = stage
        self.bot_difficulty = 0.5

class MockAgent:
    def __init__(self, id):
        self.id = id
    def state_dict(self):
        return {"dummy": torch.tensor([1.0])}

def test_checkpoint_saving():
    print("--- Testing Checkpoint Saving ---")
    
    # 1. Setup Mock Config
    config = TrainingConfig(max_checkpoints_to_keep=3)
    
    # 2. Setup Mock Trainer (Minimal)
    trainer = PPOTrainer.__new__(PPOTrainer) # Bypass init
    trainer.config = config
    trainer.env = MockEnv(stage=1) # Stage 1
    trainer.generation = 0
    trainer.logger_callback = lambda x: print(f"LOG: {x}")
    trainer.device = "cpu"
    
    # Mock Population
    trainer.population = [
        {'id': 0, 'agent': MockAgent(0), 'laps_score': 0, 'checkpoints_score': 0, 'reward_score': 0, 'ema_efficiency': 0, 'ema_consistency': 0, 'ema_wins': 0, 'novelty_score': 0},
        {'id': 1, 'agent': MockAgent(1), 'laps_score': 0, 'checkpoints_score': 0, 'reward_score': 0, 'ema_efficiency': 0, 'ema_consistency': 0, 'ema_wins': 0, 'novelty_score': 0}
    ]
    
    # Mock Components
    trainer.rms_self = MagicMock()
    trainer.rms_self.state_dict.return_value = {}
    trainer.rms_ent = MagicMock()
    trainer.rms_ent.state_dict.return_value = {}
    trainer.rms_cp = MagicMock()
    trainer.rms_cp.state_dict.return_value = {}
    trainer.league = MagicMock()
    
    # Clean up test area
    test_stage_dir = "data/stage_1"
    if os.path.exists(test_stage_dir):
        shutil.rmtree(test_stage_dir)
    
    # 3. Simulate Saving
    print("Simulating Save: Gen 0, 1, 2, 3 (Limit 3)...")
    
    # Gen 0
    trainer.generation = 0
    trainer.save_generation()
    time.sleep(0.1)
    
    # Gen 1
    trainer.generation = 1
    trainer.save_generation()
    time.sleep(0.1)

    # Gen 2
    trainer.generation = 2
    trainer.save_generation()
    time.sleep(0.1)
    
    # Verify: Should have 0, 1, 2
    gens = os.listdir(test_stage_dir)
    print(f"Dirs after 3 saves: {gens}")
    assert len(gens) == 3
    assert "gen_0" in gens
    assert "gen_2" in gens
    
    # Gen 3 (Should trigger prune of Gen 0)
    print("Saving Gen 3 (Should prune Gen 0)...")
    trainer.generation = 3
    trainer.save_generation()
    
    # Verify
    gens = os.listdir(test_stage_dir)
    print(f"Dirs after 4th save: {gens}")
    assert len(gens) == 3
    assert "gen_0" not in gens
    assert "gen_3" in gens
    
    # 4. Test Date-Based Pruning (Not just Name)
    # We have 1, 2, 3. 
    # Let's touch Gen 2 to make it NEWER than Gen 3.
    # Then Save Gen 4. 
    # Logic sorts by mtime (Oldest First).
    # Order was: 1 (Oldest), 2, 3 (Newest).
    # If we touch 1 to make it Newest.
    # Order becomes: 2 (Oldest), 3, 1 (Newest).
    # Pruning (Keep 3) -> Save 4 -> Now 4 exist: [2, 3, 1, 4]. 
    # Delete Oldest -> 2 should go.
    
    print("Touching Gen 1 to make it newest...")
    path_1 = os.path.join(test_stage_dir, "gen_1")
    os.utime(path_1, None) # Touch
    
    time.sleep(1.1) 
    
    # Current Mtimes:
    # 2: Old
    # 3: Mid
    # 1: New
    
    print("Saving Gen 4 (Should prune Gen 2)...")
    trainer.generation = 4
    trainer.save_generation()
    
    gens = os.listdir(test_stage_dir)
    print(f"Dirs after manipulation: {gens}")
    assert "gen_2" not in gens # 2 was oldest
    assert "gen_1" in gens # 1 was kept because it was updated
    assert len(gens) == 3
    
    print("SUCCESS: Checkpoint Logic Verified!")

if __name__ == "__main__":
    test_checkpoint_saving()
