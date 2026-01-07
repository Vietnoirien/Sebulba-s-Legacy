import unittest
from unittest.mock import MagicMock
from training.curriculum.stages import InterceptStage
from config import CurriculumConfig
from simulation.env import STAGE_TEAM

class TestBlockerGraduation(unittest.TestCase):
    def test_graduation_logic(self):
        config = CurriculumConfig()
        stage = InterceptStage(config)
        
        # Mock Trainer
        trainer = MagicMock()
        trainer.env.bot_difficulty = 0.85
        trainer.curriculum_mode = "auto"
        
        # Setup Metrics for Graduation (> 60% Denial)
        # Recent Episodes = 1000
        # Timeouts = 700 (70% Denial)
        trainer.env.stage_metrics = {
            "recent_episodes": 1000,
            "recent_games": 300, # 1000 - 300 = 700 timeouts
            "recent_wins": 0
        }
        
        # Run Update
        next_stage, reason = stage.update(trainer)
        
        
        # Verify
        print(f"Update Result: {next_stage}, {reason}")
        
        # Should graduate to STAGE_TEAM
        self.assertEqual(next_stage, STAGE_TEAM)
        self.assertIn("Denial Rate", reason)
        
        # Verify Difficulty Update
        # Default mock 0.85 -> Should become team_start_difficulty (0.6 default in test, need to ensuring config has it)
        # Config is real CurriculumConfig, so it has .team_start_difficulty = 0.6
        self.assertAlmostEqual(trainer.env.bot_difficulty, 0.6)

if __name__ == '__main__':
    unittest.main()
