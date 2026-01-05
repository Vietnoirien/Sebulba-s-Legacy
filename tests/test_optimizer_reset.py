
import torch
import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.optimizers import VectorizedAdam

class TestOptimizerReset(unittest.TestCase):
    def test_reset_state(self):
        # Setup
        pop_size = 5
        param_dim = 10
        params = {'w': torch.randn(pop_size, param_dim)}
        lrs = torch.ones(pop_size) * 0.01
        
        opt = VectorizedAdam(params, lrs)
        
        # Simulate Training Step (Populate State)
        grads = {'w': torch.randn(pop_size, param_dim)}
        opt.step(grads)
        
        # Verify State is Non-Zero
        self.assertEqual(opt.step_count, 1)
        self.assertFalse(torch.all(opt.exp_avg['w'] == 0))
        self.assertFalse(torch.all(opt.exp_avg_sq['w'] == 0))
        
        print("Before Reset: Step Count 1, State Non-Zero.")
        
        # Reset
        opt.reset_state()
        
        # Verify Reset
        self.assertEqual(opt.step_count, 0)
        self.assertTrue(torch.all(opt.exp_avg['w'] == 0))
        self.assertTrue(torch.all(opt.exp_avg_sq['w'] == 0))
        
        print("After Reset: Step Count 0, State Zero.")

if __name__ == '__main__':
    unittest.main()
