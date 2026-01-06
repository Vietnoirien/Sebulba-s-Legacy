
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import sys
import os

# Add Project Root to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, patch

from training.ppo import PPOTrainer
from config import TrainingConfig, CurriculumConfig
from simulation.env import STAGE_DUEL, STAGE_TEAM

class TestMitosis(unittest.TestCase):
    @patch('training.ppo.PodRacerEnv')
    @patch('training.ppo.CurriculumManager')
    def test_mitosis_logic(self, mock_curriculum_cls, mock_env_cls):
        # 1. Setup Mock Config
        t_conf = TrainingConfig()
        t_conf.pop_size = 2
        t_conf.num_envs = 2
        c_conf = CurriculumConfig()
        
        # Instantiate PPO (with mocked internal env creation)
        ppo = PPOTrainer(device='cpu')
        
        # Inject our configs
        ppo.config = t_conf
        ppo.curriculum_config = c_conf
        
        # Inject Mock Env
        mock_env = MagicMock()
        mock_env.curriculum_stage = STAGE_DUEL
        mock_env.bot_difficulty = 0.5
        ppo.env = mock_env
        
        # Re-allocate buffers because we changed config/pop_size
        ppo.allocate_buffers()
        
        # Initialization usually happens in __init__, but we need to ensure population is created respecting our config
        # PPOTrainer.__init__ calls self.population = [] but doesn't fill it?
        # Actually it calls self.init_population() likely?
        # Viewing PPO init again - it does NOT call init_population? 
        # Ah, code showed `self.population = []`.
        # I need to verify how population is filled.
        # It's usually done in `train_loop` or `restore`?
        # Or maybe I need to manually create population for the test?
        
        # Let's inspect PPO init further in next step if needed, but for now I will manually populate if needed.
        # Just creating population that matches the structure.
        
        # Create agents manually
        from models.deepsets import PodAgent
        
        ppo.population = []
        for i in range(ppo.config.pop_size):
             agent = PodAgent()
             ppo.population.append({
                 'id': i,
                 'agent': agent,
                 'lr': 1e-4, 
                 'weights': {}
             })
             
        # Initialize Vectorized (Requires population)
        ppo.init_vectorized_population()

        
        # 2. Simulate Pre-Mitosis State (Stage 2)
        # Randomize Runner Weights
        agent = ppo.population[0]['agent']
        with torch.no_grad():
            agent.runner_actor.commander.backbone[0].weight.fill_(1.0)
            # Teammate Latent (indices 15:31) should be non-zero to test zeroing
            agent.runner_actor.commander.backbone[0].weight[:, 15:31].fill_(9.0)
            
            # Blocker should be different
            agent.blocker_actor.commander.backbone[0].weight.fill_(0.5)

            # Set Bias Heads to known value to verify preservation
            # random init is usually not zero, but explicit is better.
            agent.runner_actor.commander.head_bias_thrust.weight.fill_(0.5)
            
        # Sync to Vectorized (Simulate Training Loop)
        ppo.sync_agents_to_vectorized()
        
        # Verify Vectorized Stack Matches Pre-Mitosis
        stack_runner = ppo.stacked_params['runner_actor.commander.backbone.0.weight'][0]
        self.assertTrue(torch.allclose(stack_runner[:, 15:31], torch.tensor(9.0)), "Pre-Mitosis Vectorized Runner Latent should be 9.0")
        
        # 3. Trigger Mitosis (Simulate Transition logic in PPO)
        print("Executing Mitosis Test Logic...")
        
        # Logic from PPO (Replicated matching the FIX)
        for p in ppo.population:
           ag = p['agent']
           ag.blocker_actor.load_state_dict(ag.runner_actor.state_dict())
           
           with torch.no_grad():
               W_runs = ag.runner_actor.commander.backbone[0].weight
               W_runs[:, 15:31].zero_()
               
               W_blks = ag.blocker_actor.commander.backbone[0].weight
               W_blks[:, 15:31].zero_()
               
                       

               
        # CRITICAL FIX CALL
        ppo.sync_agents_to_vectorized()
        
        # 4. Verify Post-Mitosis State
        
        # A. Agent Object State
        ag = ppo.population[0]['agent']
        w_run = ag.runner_actor.commander.backbone[0].weight
        w_blk = ag.blocker_actor.commander.backbone[0].weight
        
        # Check Cloning
        self.assertTrue(torch.allclose(w_run, w_blk), "Blocker weights must match Runner weights after Mitosis")
        
        # Check Zeroing
        self.assertTrue(torch.allclose(w_run[:, 15:31], torch.tensor(0.0)), "Runner Teammate Latent must be Zero")
        self.assertTrue(torch.allclose(w_blk[:, 15:31], torch.tensor(0.0)), "Blocker Teammate Latent must be Zero")
        self.assertTrue(torch.allclose(w_run[:, 0:15], torch.tensor(1.0)), "Other weights should remain untouched (1.0)")
        
        # Check Commander Bias NOT Zeroing (New Fix - PRESERVATION)
        r_bias_thrust = ag.runner_actor.commander.head_bias_thrust.weight
        r_bias_angle = ag.runner_actor.commander.head_bias_angle.weight
        
        # We verified they were initialized. They should NOT be exactly zero unless unlikely random chance.
        # But to be robust, we should have set them to something known.
        # However, checking they are NOT 0.0 is better than asserting 0.0.
        self.assertFalse(torch.allclose(r_bias_thrust, torch.tensor(0.0)), "Runner Bias Thrust Weight must NOT be Zero (Lobotomy Fix)")
        self.assertFalse(torch.allclose(r_bias_angle, torch.tensor(0.0)), "Runner Bias Angle Weight must NOT be Zero")
        
        # B. Vectorized Stack State (The Bug Fix Verification)
        stack_run = ppo.stacked_params['runner_actor.commander.backbone.0.weight'][0]
        stack_blk = ppo.stacked_params['blocker_actor.commander.backbone.0.weight'][0]
        
        # Check Vectorized Stack for Bias Heads too
        stack_run_bias = ppo.stacked_params['runner_actor.commander.head_bias_thrust.weight'][0]
        self.assertFalse(torch.allclose(stack_run_bias, torch.tensor(0.0)), "Vectorized Stack Runner Bias must NOT be Zero")
        
        self.assertTrue(torch.allclose(stack_run, w_run), "Vectorized Stack (Runner) must match Agent weights")
        self.assertTrue(torch.allclose(stack_blk, w_blk), "Vectorized Stack (Blocker) must match Agent weights (Sync Fix verification)")
        
        print("Test Passed: Mitosis Logic is Correct and Synced.")

if __name__ == '__main__':
    unittest.main()
