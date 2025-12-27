
import torch
import sys
import os
import json
import random
from unittest.mock import MagicMock, patch

# Mock Env to verify signature
from simulation.env import PodRacerEnv

# Test 1: Verify Env Signature
def test_env_signature():
    print("Testing Env step() signature...")
    env = PodRacerEnv(128, device='cpu')
    actions = torch.zeros((128, 4, 4))
    weights = torch.zeros((128, 11))
    
    try:
        # Check if team_spirit is accepted
        env.step(actions, reward_weights=weights, tau=0.5, team_spirit=0.5)
        print("✅ Env.step accepted team_spirit!")
    except TypeError as e:
        print(f"❌ Env.step rejected team_spirit: {e}")
        sys.exit(1)

# Test 2: Verify League Persistence & Logic
from training.self_play import LeagueManager
def test_league_manager():
    print("Testing LeagueManager Persistence & Logic...")
    
    # Clean up previous tests
    if os.path.exists("data/payoff.json"):
        os.remove("data/payoff.json")
    
    league = LeagueManager()
    
    # 1. Test Persistence
    league.update_match_result("agent_A", "agent_B", 1.0)
    
    if not os.path.exists("data/payoff.json"):
        print("❌ Failed to create payoff.json")
        sys.exit(1)
        
    with open("data/payoff.json", 'r') as f:
        data = json.load(f)
        if "agent_A:agent_B" in data:
            print("✅ Payoff Matrix Saved Correctly")
        else:
            print(f"❌ Payoff Matrix Metadata missing: {data}")
            sys.exit(1)
            
    # 2. Test Reload
    league2 = LeagueManager()
    if "agent_A:agent_B" in league2.payoff:
        print("✅ Payoff Matrix Loaded Correctly")
    else:
        print("❌ Failed to reload Payoff Matrix")
        sys.exit(1)

    # 3. Test Sampling Logic
    # Populate Registry
    league.registry = [
        {'id': 'old', 'step': 1, 'path': 'old.pt', 'elo': 1000},
        {'id': 'new', 'step': 2, 'path': 'new.pt', 'elo': 1200}
    ]
    
    # Force Exploiter (10% chance)
    # We patch random.random to return 0.05
    with patch('random.random', return_value=0.05):
        try:
            choice = league.sample_opponent(mode='pfsp')
            if choice == 'new.pt':
                 print("✅ Implicit Exploiter Logic: Selected Latest Gen (Step 2)")
            else:
                 print(f"❌ Implicit Exploiter Logic Failed: Got {choice}, Expected 'new.pt'")
        except Exception as e:
            print(f"❌ Error during sampling: {e}")

    print("League Logic Verification Complete.")

if __name__ == "__main__":
    test_env_signature()
    test_league_manager()
