import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from simulation.env import PodRacerEnv
from config import *

def check_map_quality():
    print("Initializing Env...")
    # Use CPU for easier debugging/printing if possible, but code might rely on CUDA if available.
    # We'll use "cuda" if available, else cpu.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a large batch to increase chance of hitting bad seeds
    try:
        env = PodRacerEnv(num_envs=1000, device=device, start_stage=STAGE_SOLO)
    except Exception as e:
        print(f"Failed to init env: {e}")
        return

    print("Checking Checkpoints...")
    
    checkpoints = env.checkpoints # [1000, 8, 2]
    num_cps = env.num_checkpoints # [1000]
    
    # 1. Check for (0,0) in ACTIVE checkpoints
    # We iterate manually to be precise
    zeros_found = 0
    min_dist_found = 99999.0
    border_violations = 0
    short_races = 0
    
    total_active_cps = 0
    
    for i in range(1000):
        n = num_cps[i].item()
        if n < MIN_CHECKPOINTS:
            short_races += 1
            
        cps = checkpoints[i, :n] # [n, 2]
        total_active_cps += n
        
        # Check (0,0)
        # Assuming map is large, exact (0,0) is highly unlikely unless bug.
        if (cps == 0).all(dim=1).any():
            print(f"Env {i}: Found (0,0) checkpoint! Active CPs: {n}")
            print(cps)
            zeros_found += 1
            
        # Check Border (2500 buffer?)
        # User complained about "near border". 
        # Map is 0..16000 x 0..9000
        BUFFER = 2000 # Current implicit buffer logic
        x = cps[:, 0]
        y = cps[:, 1]
        
        if (x < BUFFER).any() or (x > WIDTH - BUFFER).any() or \
           (y < BUFFER).any() or (y > HEIGHT - BUFFER).any():
               border_violations += 1
               # print(f"Env {i}: Border Violation.")
               
        # Check Pairwise Distances
        # simplified check: dist between ALL pairs
        if n > 1:
            # Expand dims to matrix
            # A: [n, 1, 2], B: [1, n, 2]
            dists = torch.cdist(cps.unsqueeze(0), cps.unsqueeze(0)).squeeze(0)
            # Mask diagonal (0)
            mask = torch.eye(n, device=device).bool()
            dists = dists.masked_fill(mask, 99999.0)
            
            curr_min = dists.min().item()
            if curr_min < min_dist_found:
                min_dist_found = curr_min
            
            if curr_min < 2500.0:
                 # print(f"Env {i}: Distance Violation {curr_min:.2f}")
                 pass

    print("--- Results ---")
    print(f"Total Envs: 1000")
    print(f"Short Races (<{MIN_CHECKPOINTS}): {short_races}")
    print(f"Zeros Found (0,0): {zeros_found}")
    print(f"Border Violations (<2000 from edge): {border_violations}")
    print(f"Global Min Pairwise Dist: {min_dist_found:.2f}")
    
    if zeros_found > 0 or short_races > 0 or min_dist_found < 2500 or border_violations > 0:
        print("FAIL: Issues detected.")
    else:
        print("SUCCESS: All maps valid.")

if __name__ == "__main__":
    check_map_quality()
