
import math

class RewardBalanceCalculator:
    def __init__(self):
        # Default Weights from Config (Proposed)
        self.weights = {
            "RW_WIN": 10000.0,
            "RW_LOSS": 2000.0,
            "RW_CHECKPOINT": 500.0,      # PROPOSED: Reduced from 2000
            "RW_CHECKPOINT_SCALE": 50.0,
            "RW_LAP": 2000.0,            # PROPOSED: New
            "RW_PROGRESS": 0.2,
            "RW_COLLISION_RUNNER": 0.5,
            "RW_COLLISION_BLOCKER": 5.0,
            "RW_STEP_PENALTY": 10.0,
            "RW_ORIENTATION": 1.0,
            "RW_WRONG_WAY": 10.0,
            "RW_COLLISION_MATE": 2.0,
            "RW_PROXIMITY": 5.0,
            "RW_MAGNET": 10.0,
            "RW_RANK": 500.0
        }
        
        self.constants = {
            "LAP_MULTIPLIER": 1.5,
            "MAX_SPEED": 800.0, # u/s
            "FPS": 60, # physics steps per sec
            "AVG_WALL_IMPACT": 400.0, # Est impulse
            "AVG_POD_IMPACT": 600.0,
            "BIG_CRASH_IMPACT": 1000.0
        }

    def print_header(self, title):
        print(f"\n{'='*10} {title} {'='*10}")

    def calc(self):
        w = self.weights
        c = self.constants
        
        events = []
        
        # 1. Navigation / Progress
        # Progress per step at max speed
        step_dist = c["MAX_SPEED"] # Velocity, roughly. Pulse is 60hz? No config doesn't say.
        # Physics step is usually smaller. Let's assume 1.0 progress unit = 1 distance unit.
        # If moving 800 u/s, and dt=1/60, moves 13.3 u/step.
        step_dist_per_tick = 800.0 / 60.0
        r_progress_step = step_dist_per_tick * w["RW_PROGRESS"]
        events.append(("Progress (Max Speed/Step)", r_progress_step, "Continuous"))
        
        # Step Penalty
        r_step = -w["RW_STEP_PENALTY"]
        events.append(("Step Penalty", r_step, "Continuous"))
        
        # Orientation (Perfect)
        # Alignment 1.0, Threshold 0.5. (1.0 - 0.5)/(0.5) = 1.0. 
        # w["RW_ORIENTATION"] * 1.0
        events.append(("Orientation (Perfect)", w["RW_ORIENTATION"], "Continuous"))
        
        # Net 'Cruising' Reward (Progress + Step + Orient)
        net_cruise = r_progress_step + r_step + w["RW_ORIENTATION"]
        events.append(("Net Cruising Reward/Step", net_cruise, "Net"))

        # 2. Checkpoints
        events.append(("Checkpoint (Base)", w["RW_CHECKPOINT"], "Discrete"))
        events.append(("Checkpoint (Streak 10)", w["RW_CHECKPOINT"] + (10 * w["RW_CHECKPOINT_SCALE"]), "Discrete"))
        
        # 3. Laps
        events.append(("Lap 1 Completion", w["RW_LAP"] * (c["LAP_MULTIPLIER"]**0), "Discrete"))
        events.append(("Lap 2 Completion", w["RW_LAP"] * (c["LAP_MULTIPLIER"]**1), "Discrete"))
        events.append(("Lap 3 Completion (Win)", w["RW_WIN"], "Terminal")) # Win reward usually overrides
        
        # 4. Combat / Interaction
        # Overtake
        events.append(("Rank Improvement (+1)", w["RW_RANK"], "Discrete"))
        events.append(("Rank Loss (-1)", -w["RW_RANK"], "Discrete"))
        
        # Collisions
        # Wall (Runner)
        # Impact * Weight
        r_wall = -c["AVG_WALL_IMPACT"] * w["RW_COLLISION_RUNNER"]
        events.append(("Collision Wall (Runner, Avg)", r_wall, "Penalty"))
        
        # Mate
        r_mate = -c["AVG_POD_IMPACT"] * w["RW_COLLISION_MATE"]
        events.append(("Collision Teammate (Avg)", r_mate, "Penalty"))
        
        # Blocker Hit (As Blocker)
        # This is a POSITIVE reward for the Blocker if they hit a Runner?
        # Typically RW_COLLISION_BLOCKER is positive?
        # Let's assume Config: RW_COLLISION_BLOCKER: 5.0
        # If I am blocker, and I hit runner: Impact * 5.0
        r_block_hit = c["BIG_CRASH_IMPACT"] * w["RW_COLLISION_BLOCKER"]
        events.append(("Blocker Impact (Big Hit)", r_block_hit, "Bonus"))

        
        # --- Visualization ---
        self.print_header("REWARD BALANCE ANALYSIS")
        print(f"{'Event':<30} | {'Value':<10} | {'Type':<10}")
        print("-" * 55)
        for name, val, ktype in events:
            print(f"{name:<30} | {val:10.2f} | {ktype:<10}")

if __name__ == "__main__":
    RewardBalanceCalculator().calc()
