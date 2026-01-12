
import torch
import torch
import numpy as np
# import matplotlib.pyplot as plt
from typing import Tuple

def calculate_zone_pressure_reward(
    blocker_pos: torch.Tensor, 
    enemy_pos: torch.Tensor, 
    cp_pos: torch.Tensor, 
    weight: float = 1.0, 
    debug: bool = False
) -> float:
    """
    Replicates the logic from env.py for RW_ZONE_PRESSURE.
    """
    # Vectors
    # Opp -> CP (Target Vector)
    dir_en_cp_raw = cp_pos - enemy_pos
    dist_en_cp = torch.norm(dir_en_cp_raw) + 1e-6
    vec_opp_cp = dir_en_cp_raw / dist_en_cp # Normalized

    # Opp -> Me (Blocker Position relative to Opponent)
    vec_opp_me = blocker_pos - enemy_pos
    dist_opp_me = torch.norm(vec_opp_me) + 1e-6
    dir_opp_me = vec_opp_me / dist_opp_me

    # Alignment (Dot Product)
    align_shepherd = (vec_opp_cp * dir_opp_me).sum()

    # Only reward positive alignment (being in front)
    shepherd_score = torch.clamp(align_shepherd, 0.0, 1.0)

    # Zone Factor: Risk = 1.0 - (dist / 5000).
    risk_factor = torch.clamp(1.0 - (dist_en_cp / 5000.0), 0.2, 1.0)

    # Reward
    r_shepherd = shepherd_score * risk_factor * weight
    
    if debug:
        print(f"DistEnCP: {dist_en_cp:.1f}, Risk: {risk_factor:.2f}, Align: {align_shepherd:.2f}, Score: {shepherd_score:.2f}, RawReward: {r_shepherd:.4f}")
        
    return r_shepherd.item()

def run_analysis():
    print("=== Analyzing Zone Pressure Reward Value ===")
    
    # Constants
    scale_weight = 1.0 # Base weight for analysis
    
    # Scenario Setup
    # Enemy at (0,0)
    enemy_pos = torch.tensor([0.0, 0.0])
    
    # Checkpoint at (3000, 0) -> Distance 3000
    cp_pos = torch.tensor([3000.0, 0.0])
    
    print(f"\nScenario: Enemy at {enemy_pos}, CP at {cp_pos} (Dist: 3000)")
    
    # Test Cases
    positions = [
        ("Perfect Block (100u Front)", torch.tensor([100.0, 0.0])),
        ("Mid Block (1500u Front)", torch.tensor([1500.0, 0.0])),
        ("Far Block (2900u Front)", torch.tensor([2900.0, 0.0])),
        ("Side (Parallel 500u)", torch.tensor([500.0, 500.0])),
        ("Behind (-500u)", torch.tensor([-500.0, 0.0])),
    ]
    
    for name, b_pos in positions:
        rew = calculate_zone_pressure_reward(b_pos, enemy_pos, cp_pos, weight=scale_weight, debug=False)
        # Calculate theoretical max per episode (assuming 300 steps of maintaining this)
        total_ep = rew * 300
        print(f"  {name:25}: Reward/Step = {rew:.4f} | Total/ Ep (~300 steps) = {total_ep:.1f}")

    print("\n--- Sensitivity to CP Distance (Risk Factor) ---")
    dists = [1000, 3000, 5000, 8000]
    for d in dists:
        cp = torch.tensor([float(d), 0.0])
        blocker = torch.tensor([100.0, 0.0]) # Always close perfect block
        rew = calculate_zone_pressure_reward(blocker, enemy_pos, cp, weight=scale_weight)
        print(f"  CP Dist {d:<5}: Step Reward = {rew:.4f} (Risk Factor Impact)")

if __name__ == "__main__":
    run_analysis()
