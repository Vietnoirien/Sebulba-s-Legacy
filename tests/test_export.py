import unittest
import torch
import os
import subprocess
from models.deepsets import PodAgent
from config import WIDTH, HEIGHT

class TestExport(unittest.TestCase):
    def setUp(self):
        self.model_path = "test_model.pt"
        self.out_path = "test_submission.py"
        
        # Create Dummy Model
        agent = PodAgent()
        torch.save(agent.state_dict(), self.model_path)
        
    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.out_path):
            os.remove(self.out_path)
            
    def test_export_script(self):
        # Run export.py via subprocess
        # Assumes export.py is in root
        cmd = [
            "/home/viet/git-perso/SPT2/.venv/bin/python", 
            "export.py", 
            "--model", self.model_path, 
            "--out", self.out_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check Success
        self.assertEqual(result.returncode, 0, f"Export failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.out_path), "Output file not created")
        
        # Verify Content
        with open(self.out_path, 'r') as f:
            content = f.read()
            
        self.assertIn(f"WIDTH = {WIDTH}", content)
        self.assertIn(f"HEIGHT = {HEIGHT}", content)
        self.assertIn("class Agent(NN):", content)
        self.assertIn("WEIGHTS =", content)

if __name__ == '__main__':
    unittest.main()
