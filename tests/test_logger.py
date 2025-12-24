
import unittest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from training.ppo import PPOTrainer

class TestLoggerCallback(unittest.TestCase):
    def test_log_callback(self):
        # Mock callback
        mock_cb = MagicMock()
        
        # Init Trainer
        trainer = PPOTrainer(device='cpu', logger_callback=mock_cb)
        
        # Trigger log
        test_msg = "Test Log Message"
        trainer.log(test_msg)
        
        # Verify
        mock_cb.assert_called_with(test_msg)
        print("Logger callback verification passed!")

if __name__ == '__main__':
    unittest.main()
