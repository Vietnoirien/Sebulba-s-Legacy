
import unittest
from unittest.mock import MagicMock
from training.ppo import PPOTrainer, STAGE_DUEL

class TestDifficultyLogic(unittest.TestCase):
    def test_difficulty_adjustment(self):
        # Mock Trainer
        trainer = MagicMock(spec=PPOTrainer)
        trainer.log = MagicMock()
        trainer.env = MagicMock()
        trainer.env.curriculum_stage = STAGE_DUEL
        trainer.failure_streak = 0
        trainer.curriculum_mode = "auto"
        
        # Bind check_curriculum to the mock instance (tricky with MagicMock)
        # Better: Instantiate real PPOTrainer but mock its internals?
        # Too heavy. Let's just create a dummy class or minimal subclass.
        pass

# Minimal Mock
class MockEnv:
    def __init__(self):
        self.curriculum_stage = STAGE_DUEL
        self.bot_difficulty = 0.5
        self.stage_metrics = {
            "recent_games": 2000,
            "recent_wins": 0
        }

class MockTrainer:
    def __init__(self):
        self.env = MockEnv()
        self.failure_streak = 0
        self.curriculum_mode = "auto"
        self.iteration = 1
    
    def log(self, msg):
        print(msg)

    # Copy the method logic here? Or import?
    # We want to test the *actual* code in ppo.py.
    # We can import existing check_curriculum.
    
    from training.ppo import PPOTrainer
    check_curriculum = PPOTrainer.check_curriculum

class TestDifficulty(unittest.TestCase):
    def test_super_turbo(self):
        trainer = MockTrainer()
        trainer.env.bot_difficulty = 0.5
        
        # Test Case 1: WR > 98% (e.g. 99%)
        trainer.env.stage_metrics['recent_games'] = 1001
        trainer.env.stage_metrics['recent_wins'] = 990 # 99%
        
        trainer.check_curriculum()
        
        self.assertAlmostEqual(trainer.env.bot_difficulty, 0.7) # 0.5 + 0.2
        print("Success: Super Turbo (+0.2)")

    def test_turbo(self):
        trainer = MockTrainer()
        trainer.env.bot_difficulty = 0.5
        
        # Test Case 2: WR > 90% (e.g. 95%)
        trainer.env.stage_metrics['recent_games'] = 1001
        trainer.env.stage_metrics['recent_wins'] = 950 # 95%
        
        trainer.check_curriculum()
        
        self.assertAlmostEqual(trainer.env.bot_difficulty, 0.6) # 0.5 + 0.1
        print("Success: Turbo (+0.1)")

    def test_normal(self):
        trainer = MockTrainer()
        trainer.env.bot_difficulty = 0.5
        
        # Test Case 3: WR > 70% (e.g. 80%)
        trainer.env.stage_metrics['recent_games'] = 1001
        trainer.env.stage_metrics['recent_wins'] = 800 # 80%
        
        trainer.check_curriculum()
        
        self.assertAlmostEqual(trainer.env.bot_difficulty, 0.55) # 0.5 + 0.05
        print("Success: Normal (+0.05)")

if __name__ == '__main__':
    unittest.main()
