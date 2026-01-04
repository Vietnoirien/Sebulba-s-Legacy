
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_REWARD_WEIGHTS

class RewardComparator:
    def __init__(self):
        self.current_weights = DEFAULT_REWARD_WEIGHTS.copy()
        
        # Proposed Weights
        self.proposed_weights = DEFAULT_REWARD_WEIGHTS.copy()
        self.proposed_weights.update({
             2: 500.0,   # RW_CHECKPOINT (Index 2)
             # RW_LAP doesn't exist in current config, but we simulated it.
             # We will handle it manually in events.
        })
        
        self.constants = {
            "LAP_MULTIPLIER": 1.5,
            "MAX_SPEED": 800.0,
            "FPS": 60,
            "AVG_WALL_IMPACT": 400.0,
            "AVG_POD_IMPACT": 600.0,
            "BIG_CRASH_IMPACT": 1000.0
        }

    def calc_events(self, weights, is_proposed=False):
        w = weights
        c = self.constants
        
        # Mapping indices to keys for readability if needed, but we used IDs in config.py
        # RW_WIN = 0
        # RW_LOSS = 1
        # RW_CHECKPOINT = 2
        # RW_CHECKPOINT_SCALE = 3
        # RW_PROGRESS = 4
        # RW_COLLISION_RUNNER = 5
        # RW_COLLISION_BLOCKER = 6
        # RW_STEP_PENALTY = 7
        # RW_ORIENTATION = 8
        # RW_WRONG_WAY = 9
        # RW_COLLISION_MATE = 10
        # RW_RANK = 13
        
        # Helper to get weight safely
        def W(idx): return w.get(idx, 0.0)
        
        vals = {}
        
        # Progress
        step_dist_per_tick = 800.0 / 60.0
        vals["Progress/Step"] = step_dist_per_tick * W(4)
        vals["Step Penalty"] = -W(7)
        vals["Net Cruise"] = vals["Progress/Step"] + vals["Step Penalty"] + W(8) # + Orient
        
        # Checkpoint
        vals["Checkpoint"] = W(2)
        vals["CP (Streak 10)"] = W(2) + (10 * W(3))
        
        # Laps
        if is_proposed:
            # Proposed Logic: Lap Reward exists
            RW_LAP = 2000.0
            vals["Lap 1"] = RW_LAP * (c["LAP_MULTIPLIER"]**0)
            vals["Lap 2"] = RW_LAP * (c["LAP_MULTIPLIER"]**1)
        else:
            # Current Logic: Lap is just a checkpoint (CP 0) + maybe Metric update
            # Environment treats CP 0 as just another CP for reward purposes currently.
            vals["Lap 1"] = W(2) # Just a standard CP reward
            vals["Lap 2"] = W(2) # Just a standard CP reward
            
        vals["Win"] = W(0)
        vals["Rank (+1)"] = W(13)
        vals["Wall Hit"] = -c["AVG_WALL_IMPACT"] * W(5)
        vals["Blocker Hit"] = c["BIG_CRASH_IMPACT"] * W(6) # Bonus for blocker
        
        return vals

    def run(self):
        curr = self.calc_events(self.current_weights, is_proposed=False)
        prop = self.calc_events(self.proposed_weights, is_proposed=True)
        
        print("\n" + "="*70)
        print(f"{'Event':<20} | {'Current':<10} | {'Proposed':<10} | {'Change':<10}")
        print("-" * 70)
        
        keys = ["Progress/Step", "Step Penalty", "Net Cruise", 
                "Checkpoint", "CP (Streak 10)", 
                "Lap 1", "Lap 2", "Win",
                "Rank (+1)", "Wall Hit", "Blocker Hit"]
                
        for k in keys:
            c_val = curr.get(k, 0.0)
            p_val = prop.get(k, 0.0)
            diff = p_val - c_val
            
            # Format
            c_str = f"{c_val:8.1f}"
            p_str = f"{p_val:8.1f}"
            d_str = f"{diff:+8.1f}"
            
            print(f"{k:<20} | {c_str} | {p_str} | {d_str}")
        print("="*70 + "\n")

if __name__ == "__main__":
    RewardComparator().run()
