
import torch
import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.normalization import RunningMeanStd

class TestRMSReset(unittest.TestCase):
    def test_reset_logic(self):
        device = 'cpu'
        mean_std = RunningMeanStd(shape=(5,), device=device)
        
        # Populate with data
        data = torch.ones((10, 5), device=device) * 10.0
        mean_std.update(data)
        
        # Verify Non-Zero
        self.assertTrue(torch.allclose(mean_std.mean, torch.tensor(10.0), atol=1e-3))
        self.assertAlmostEqual(mean_std.count.item(), 10.0 + 1e-4, places=4)
        
        print(f"Before Reset: Mean={mean_std.mean}")
        
        # Reset
        mean_std.reset()
        
        # Verify Reset
        self.assertTrue(torch.all(mean_std.mean == 0.0))
        self.assertTrue(torch.all(mean_std.var == 1.0))
        self.assertTrue(mean_std.count.item() < 1.0) # Should be epsilon
        
        print("After Reset: Mean=0, Var=1. Verified.")

if __name__ == '__main__':
    unittest.main()
