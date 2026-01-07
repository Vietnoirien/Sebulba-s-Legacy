
import unittest
import torch
import numpy as np
from simulation.env import PodRacerEnv, STAGE_TEAM
from training.curriculum.manager import CurriculumManager
from training.ppo import PPOTrainer
from config import EnvConfig, CurriculumConfig

class TestTeamSpiritAndBlocker(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.env_config = EnvConfig()
        self.curr_config = CurriculumConfig()
        
    def test_blocker_racing_reward_masked(self):
        """Verify Blockers get zero reward for checkpoitns in Stage 4."""
        # Fix init signature: num_envs, device
        env = PodRacerEnv(num_envs=2, device="cpu")
        env.set_stage(STAGE_TEAM, self.env_config)
        
        # Setup: Pod 0 is Runner, Pod 1 is Blocker (Team 0)
        # Manually force is_runner
        env.is_runner[:] = False
        env.is_runner[:, 0] = True # Pod 0 Runner
        env.is_runner[:, 1] = False # Pod 1 Blocker
        
        # Force checkpoint pass for both
        pass_idx = torch.tensor([0], device=self.device) # Only env 0 passed
        
        # We need to simulate the 'step' logic partially or mock it.
        # But 'step' is complex. Easier to run a step where we force physics to cross CP.
        
        # Let's mock the internal variables right before reward calc? 
        # No, integration test is better.
        # Move pods through CP 1.
        
        # Reset
        env.reset()
        env.is_runner[:] = False
        env.is_runner[:, 0] = True
        env.is_runner[:, 1] = False 
        
        # Teleport to CP 1
        cp1_pos = env.checkpoints[0, 1]
        env.physics.pos[0, 0] = cp1_pos + torch.tensor([10.0, 0.0]) # Just past
        env.physics.pos[0, 1] = cp1_pos + torch.tensor([10.0, 1000.0]) # Offset Y by 1000 (Radius 400->800 safety)
        
        # Set next_cp to 1
        env.next_cp_id[:] = 1
        
        # Velocity forward
        env.physics.vel[:] = torch.tensor([1000.0, 0.0])
        
        # Step
        # Action: 0 (No op). Shape [B, 4, 4] (Throttle, Steer, Shield, Boost)
        actions = torch.zeros((2, 4, 4), device=self.device)
        
        # Create Dummy Reward Weights [2, 16] (assuming 16 reward types)
        reward_weights = torch.ones((2, 16), device=self.device)
        # Set RW_CHECKPOINT (index 2 usually) to a known high value
        reward_weights[:, 2] = 500.0
        
        # step() returns (rewards, dones, infos)
        rewards, dones, infos = env.step(actions, reward_weights)
        
        # Check Rewards
        runner_rew = rewards[0, 0].item()
        blocker_rew = rewards[0, 1].item()
        
        print(f"Runner Reward: {runner_rew}")
        print(f"Blocker Reward: {blocker_rew}")
        
        self.assertGreater(runner_rew, 100.0, "Runner should get CP reward")
        self.assertLess(abs(blocker_rew), 10.0, "Blocker should get ~0 reward (only small penalties/noise)")
        
    def test_team_spirit_annealing(self):
        """Verify Team Spirit calculates correctly based on Evolution Steps."""
        # Mock Trainer
        class MockTrainer:
            def __init__(self):
                self.team_spirit = 0.0 # Start
                self.population = [{'ema_wins': 0.0}]
                self.leader_idx = 0
                self.env = type('obj', (object,), {'curriculum_stage': STAGE_TEAM})
                
        trainer = MockTrainer()
        manager = CurriculumManager(self.curr_config)
        manager.current_stage_id = STAGE_TEAM
        
        # 1. Verify NO Change on regular update (Iteration)
        spirit = manager.update_team_spirit(trainer)
        print(f"Iter check -> Spirit {spirit}")
        self.assertAlmostEqual(spirit, 0.0, msg="Should NOT auto-increment on update_team_spirit")
        
        # 2. Verify +0.01 on Evolution Step
        manager.on_evolution_step(trainer)
        print(f"Evol 1 -> Spirit {trainer.team_spirit}")
        self.assertAlmostEqual(trainer.team_spirit, 0.01, msg="Should increment by 0.01 on evolution")
        
        # 3. Verify Accumulation
        manager.on_evolution_step(trainer)
        print(f"Evol 2 -> Spirit {trainer.team_spirit}")
        self.assertAlmostEqual(trainer.team_spirit, 0.02)
        
        # 4. Saturation Test
        trainer.team_spirit = 0.995
        manager.on_evolution_step(trainer)
        print(f"Evol X (0.995) -> Spirit {trainer.team_spirit}")
        self.assertAlmostEqual(trainer.team_spirit, 1.0)
        
        manager.on_evolution_step(trainer)
        print(f"Evol Y (1.0) -> Spirit {trainer.team_spirit}")
        self.assertAlmostEqual(trainer.team_spirit, 1.0)

if __name__ == '__main__':
    unittest.main()
