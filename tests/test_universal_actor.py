
import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.deepsets import PodAgent

class TestUniversalActor(unittest.TestCase):
    def test_forward_pass_universal(self):
        """
        Verifies that PodAgent (Universal) accepts inputs and returns correct shapes.
        """
        batch_size = 4
        hidden_dim = 128
        
        agent = PodAgent(hidden_dim=hidden_dim)
        
        # Dimensions
        # Self(15), Team(13), Enemy(13), NextCP(6)
        # Check env.py for exact dims if unsure, but deepsets.py defaults are 15, 13, 13
        
        self_obs = torch.randn(batch_size, 15)
        teammate_obs = torch.randn(batch_size, 13)
        # Enemy Obs: DeepSets usually expects [B, N_enemies, 13] or [B*N, 13] is handled?
        # In deepsets.py CommanderNet encoder takes [B, 13] for teammate
        # And [B, N, 13] for enemies?
        # deepsets.py: 
        #   enc_en = self.encoder(enemy_obs) # [B, N, 16]
        #   env_ctx, _ = torch.max(enc_en, dim=1)
        
        enemy_obs = torch.randn(batch_size, 3, 13) # 3 enemies
        next_cp_obs = torch.randn(batch_size, 6)
        
        # Mock 'is_runner' at index 11 of self_obs
        # Set half as runner, half blocker
        self_obs[0, 11] = 1.0
        self_obs[1, 11] = 1.0
        self_obs[2, 11] = 0.0
        self_obs[3, 11] = 0.0
        
        # Forward
        action, log_prob, entropy, value = agent.get_action_and_value(self_obs, teammate_obs, enemy_obs, next_cp_obs)
        
        # Check Shapes
        # Action: [B, 4] (Thrust, Angle, Shield, Boost - last 2 are discrete mapped to float?)
        # Let's check get_action logic.
        # It calls dist_cont.sample() -> [B, 2]
        # dist_shield.sample() -> [B]
        # returns torch.cat([ac, s, b], dim=1) -> [B, 4]
        
        self.assertEqual(action.shape, (batch_size, 4))
        self.assertEqual(value.shape, (batch_size, 1))
        self.assertEqual(log_prob.shape, (batch_size,))
        self.assertEqual(entropy.shape, (batch_size,))
        
        print(f"Action Sample: {action[0]}")
        # Verify gradients
        loss = value.mean() + log_prob.mean()
        loss.backward()
        
        # Check if role embedding received gradients
        self.assertIsNotNone(agent.actor.role_embedding.weight.grad)
        print("Role Embedding Gradients Flowing OK.")

if __name__ == '__main__':
    unittest.main()
