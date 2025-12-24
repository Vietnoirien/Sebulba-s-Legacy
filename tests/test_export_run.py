
import unittest
import torch
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from models.deepsets import PodAgent
from export import export_model

class TestExportRun(unittest.TestCase):
    def setUp(self):
        self.test_model_path = "tests/test_model.pt"
        self.output_path = "tests/submission_test.py"
        self.agent = PodAgent(hidden_dim=256)
        torch.save(self.agent.state_dict(), self.test_model_path)

    def tearDown(self):
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_export_generation(self):
        """Test if export generates a file and it has content."""
        export_model(self.test_model_path, self.output_path)
        
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)
        
        # Verify content
        with open(self.output_path, 'r') as f:
            content = f.read()
            self.assertIn("HIDDEN_DIM = 256", content)
            self.assertIn("class Agent(NN):", content)

    def test_exported_code_runnable(self):
        """Test if the exported code can be imported and the Agent class instantiated."""
        export_model(self.test_model_path, self.output_path)
        
        # Import the generated file
        import importlib.util
        spec = importlib.util.spec_from_file_location("submission_test", self.output_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Instantiate Agent
        agent = module.Agent(module.WEIGHTS, module.SCALE)
        
        # Creates dummy inputs matching the forward signature
        # Check forward signature in template: forward(self_obs, entity_obs, cp_obs)
        # self_obs: 14
        # entity_obs: [ [13]*3 ] ?? In template: for ent in entity_obs: ... ent[c]
        # In export template Agent.forward -> entity_obs is list of lists
        
        self_obs = [0.0] * 14
        entity_obs = [[0.0]*13 for _ in range(3)] 
        cp_obs = [0.0] * 6
        
        # Run forward
        action = agent.forward(self_obs, entity_obs, cp_obs)
        
        # Check output [thrust, angle, shield, boost]
        self.assertEqual(len(action), 4)
        self.assertTrue(0 <= action[0] <= 1) # Thrust sigmoid
        self.assertTrue(-1 <= action[1] <= 1) # Angle tanh
        self.assertIn(action[2], [0, 1])
        self.assertIn(action[3], [0, 1])

if __name__ == '__main__':
    unittest.main()
