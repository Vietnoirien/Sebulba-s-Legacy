
import unittest
import os
import shutil
import json
import torch
import sys

# Append root
sys.path.append(os.getcwd())

from training.self_play import LeagueManager

class TestLeaguePreservation(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_league_data"
        self.checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        self.league_store_dir = os.path.join(self.test_dir, "league_store")
        
        # Override Constants in self_play is tricky since they are global constants.
        # We will Mock them or set up the environment to point to test_dir BEFORE importing if possible?
        # No, they are imported. We must patch them.
        pass
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_copy_behavior(self):
        # Setup paths
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.league_store_dir, exist_ok=True)
        
        dummy_agent_path = os.path.join(self.checkpoints_dir, "gen_100_agent_5.pt")
        with open(dummy_agent_path, "w") as f:
            f.write("dummy content")
            
        # Patch Constants
        import training.self_play as sp
        
        original_league_dir = sp.LEAGUE_DIR
        original_league_file = sp.LEAGUE_FILE
        
        sp.LEAGUE_DIR = self.league_store_dir
        sp.LEAGUE_FILE = os.path.join(self.test_dir, "league.json")
        
        try:
            manager = LeagueManager()
            
            # Register Agent (simulate high win rate)
            manager.register_agent(
                name="gen_100_agent_5",
                path=dummy_agent_path,
                step=100,
                metrics={"wins_ema": 0.9}
            )
            
            # 1. Verify League Store has the file
            expected_store_path = os.path.join(self.league_store_dir, "gen_100_agent_5.pt")
            self.assertTrue(os.path.exists(expected_store_path), "File was not copied to league store")
            
            # 2. Verify Registry points to store path
            with open(sp.LEAGUE_FILE, "r") as f:
                reg = json.load(f)
                entry = reg[0]
                self.assertEqual(entry["path"], expected_store_path)
                
            # 3. Simulate Deletion of original (Transient)
            os.remove(dummy_agent_path)
            
            # 4. Verify League Store file still exists
            self.assertTrue(os.path.exists(expected_store_path), "League file vanished after original deletion!")
            
        finally:
            sp.LEAGUE_DIR = original_league_dir
            sp.LEAGUE_FILE = original_league_file

if __name__ == '__main__':
    unittest.main()
