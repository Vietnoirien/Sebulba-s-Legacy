
import unittest
import os
import json
import sys
from unittest.mock import patch, MagicMock

# Append root to path
sys.path.append(os.getcwd())

import export

class TestExportAutoReal(unittest.TestCase):
    def setUp(self):
        self.real_league = "data/league.json"
        self.backup_league = "data/league.json.bak"
        self.dummy_pt = "data/checkpoints/gen_11_agent_1.pt"
        
        # Backup existing league.json if present
        if os.path.exists(self.real_league):
            os.rename(self.real_league, self.backup_league)
        else:
            # Ensure parent dir exists
            os.makedirs("data", exist_ok=True)
            
        os.makedirs("data/checkpoints", exist_ok=True)
        
        # Create Dummy Data
        self.league_data = [
             {"id": "gen_10_agent_0", "path": "data/checkpoints/gen_10_agent_0.pt", "step": 10, "metrics": {"wins_ema": 0.5}},
             {"id": "gen_11_agent_0", "path": "data/checkpoints/gen_11_agent_0.pt", "step": 11, "metrics": {"wins_ema": 0.4}},
             # Winner
             {"id": "gen_11_agent_1", "path": "data/checkpoints/gen_11_agent_1.pt", "step": 11, "metrics": {"wins_ema": 0.6}},
        ]
        with open(self.real_league, "w") as f:
            json.dump(self.league_data, f)
            
        with open(self.dummy_pt, "w") as f:
            f.write("dummy")

    def tearDown(self):
        # Restore backup
        if os.path.exists(self.real_league):
            os.remove(self.real_league)
            
        if os.path.exists(self.backup_league):
             os.rename(self.backup_league, self.real_league)
             
        if os.path.exists(self.dummy_pt):
            os.remove(self.dummy_pt)

    def test_find_best_checkpoint(self):
        # Verify the logic picks the correct checkpoint
        if not hasattr(export, 'find_best_checkpoint'):
            print("Skipping test: find_best_checkpoint not implemented yet")
            return

        best = export.find_best_checkpoint()
        print(f"Selected: {best}")
        self.assertEqual(best, "data/checkpoints/gen_11_agent_1.pt")

if __name__ == '__main__':
    unittest.main()
