
import unittest
import torch
from simulation.env import PodRacerEnv, STAGE_TEAM, STAGE_SOLO
from config import EnvConfig

class TestRoleSwitching(unittest.TestCase):
    def test_blocker_becomes_runner_when_ahead(self):
        """
        Verify that in Stage 4, if Pod 1 (Blocker) gets ahead of Pod 0 (Runner), 
        roles SWAP. This confirms the 'Runner Envy' behavior.
        """
        config = EnvConfig()
        env = PodRacerEnv(num_envs=2, device="cpu")
        env.set_stage(STAGE_TEAM, config, reset_env=True)
        
        # Initial State: 
        # Pod 0 and Pod 1 are at Start.
        # Typically Index 0 is Runner initially due to tie-break.
        print(f"Initial Roles: Pod 0 Runner? {env.is_runner[0, 0]}")
        
        self.assertTrue(env.is_runner[0, 0], "Pod 0 should start as Runner")
        self.assertFalse(env.is_runner[0, 1], "Pod 1 should start as Blocker")
        
        # Teleport Pod 1 (Blocker) AHEAD of Pod 0.
        # Pod 0 at 0.0 distance (start)
        # Pod 1 at CP 1 (Passed CP 0)
        
        # Simulate Checkpoint Pass for Pod 1
        env.next_cp_id[0, 1] = 1
        env.prev_dist[0, 1] = 1000.0 # Further ahead
        
        # Simulate Pod 0 still at Start
        env.next_cp_id[0, 0] = 1 # Also aimed at 1
        env.prev_dist[0, 0] = 5000.0 # Further away
        
        # Force Update Roles
        env_ids = torch.tensor([0], device=env.device)
        env._update_roles(env_ids)
        
        print(f"New Roles: Pod 0 Runner? {env.is_runner[0, 0]}")
        print(f"New Roles: Pod 1 Runner? {env.is_runner[0, 1]}")
        
        # [FIXED BEHAVIOR] Roles should NO LONGER flip.
        self.assertTrue(env.is_runner[0, 0], "Pod 0 should REMAIN Runner (Fixed Role)")
        self.assertFalse(env.is_runner[0, 1], "Pod 1 should REMAIN Blocker (Fixed Role)")

if __name__ == '__main__':
    unittest.main()
