
import numpy as np

def calculate_trajectory_reward(distance, velocity, weights):
    """
    Simulates a linear trajectory to a checkpoint.
    """
    # 1. Physics Simulation (Simple)
    # Steps needed = Distance / Velocity
    # Assuming constant velocity for simplicity
    
    steps = int(np.ceil(distance / velocity))
    
    # 2. Reward Accumulation
    total_reward = 0.0
    
    # Unpack Weights
    w_cp = weights["checkpoint"]
    w_vel = weights["velocity"]
    w_step = weights["step_penalty"]
    
    # Per Step
    for _ in range(steps):
        # A. Velocity Reward
        # RW_VELOCITY * (Velocity . TargetDir)
        # Here Velocity aligned with TargetDir
        # But wait, original code: 
        # rewards += vel_proj * w_velocity
        # Scale? 
        # In env.py: v_scaled = vel_proj * scale
        # So it's raw scalar reward per step.
        v_reward = velocity * w_vel
        
        # B. Step Penalty
        # rewards -= w_step * alpha (assume alpha=1.0 for worst case/end of timeout)
        s_penalty = w_step * 1.0 
        
        total_reward += (v_reward - s_penalty)
        
    # C. Checkpoint Reward (End)
    total_reward += w_cp
    
    return total_reward, steps

def run_scenario(name, weights):
    print(f"--- Scenario: {name} ---")
    print(f"Weights: {weights}")
    
    DIST = 4000.0 # Standard CP distance
    
    # Agent A: Fast (800.0 max speed)
    r_fast, steps_fast = calculate_trajectory_reward(DIST, 800.0, weights)
    
    # Agent B: Slow (200.0 cruising speed)
    r_slow, steps_slow = calculate_trajectory_reward(DIST, 200.0, weights)
    
    print(f"Fast Agent (800u/s) | Steps: {steps_fast} | Total Reward: {r_fast:.2f}")
    print(f"Slow Agent (200u/s) | Steps: {steps_slow} | Total Reward: {r_slow:.2f}")
    
    delta = r_fast - r_slow
    print(f"Advantage (Fast - Slow): {delta:.2f}")
    
    # Metric: Points per Step Saved
    steps_saved = steps_slow - steps_fast
    if steps_saved > 0:
        pps = delta / steps_saved
        print(f"Incentive: {pps:.2f} points per step saved")
    else:
        print("Incentive: N/A")
    print("")

def main():
    # 1. Current Baseline
    current_weights = {
        "checkpoint": 4000.0,
        "velocity": 0.05,
        "step_penalty": 5.0
    }
    run_scenario("Current Baseline", current_weights)
    
    # 2. Proposed (Plan A)
    # V=0.5, Step=15.0
    proposed_weights = {
        "checkpoint": 4000.0,
        "velocity": 0.5,
        "step_penalty": 15.0
    }
    run_scenario("Proposed Plan", proposed_weights)

    # 3. High Urgency (Plan B - My Revision)
    # V=0.5, Step=25.0
    revised_weights = {
        "checkpoint": 4000.0,
        "velocity": 0.5,
        "step_penalty": 25.0
    }
    run_scenario("High Urgency", revised_weights)
    
    # 4. Extreme Urgency (Standard RL)
    # CP=2000, V=1.0, Step=50.0
    # Trying to make time cost dominant
    extreme_weights = {
        "checkpoint": 2000.0,
        "velocity": 1.0,
        "step_penalty": 50.0
    }
    run_scenario("Extreme Urgency", extreme_weights)

if __name__ == "__main__":
    main()
