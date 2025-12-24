
import torch
import unittest
from simulation.env import PodRacerEnv, DEFAULT_REWARD_WEIGHTS, RW_WIN, RW_LOSS
from config import STAGE_SOLO

class TestVectorizedRewards(unittest.TestCase):
    def test_diverse_rewards(self):
        # Create env with 8 envs (small batch)
        env = PodRacerEnv(num_envs=8, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare Reward Weights Tensor
        weights = torch.zeros((8, 10), device=env.device)
        
        # Fill with default first
        for k,v in DEFAULT_REWARD_WEIGHTS.items():
            weights[:, k] = v
            
        # Modify Env 0 to have ZERO win reward
        weights[0, RW_WIN] = 0.0
        # Modify Env 1 to have HUGE win reward
        weights[1, RW_WIN] = 100000.0
        
        # Manually trigger a "Win" condition
        # Set laps to MAX for pod 0 in env 0 and 1
        env.laps[0, 0] = 3
        env.laps[1, 0] = 3
        
        # Step with no actions (just to trigger game logic)
        actions = torch.zeros((8, 4, 4), device=env.device)
        rewards, dones = env.step(actions, reward_weights=weights)
        
        # Check rewards
        # Env 0 Pod 0 should have ~0 reward (minus penalties?)
        # Env 1 Pod 0 should have ~100000 reward
        
        r0 = rewards[0, 0].item()
        r1 = rewards[1, 0].item()
        
        print(f"Env 0 Reward: {r0}")
        print(f"Env 1 Reward: {r1}")
        
        self.assertLess(r0, 1000.0, "Env 0 should have low reward")
        self.assertGreater(r1, 90000.0, "Env 1 should have high reward")

if __name__ == '__main__':
    unittest.main()
