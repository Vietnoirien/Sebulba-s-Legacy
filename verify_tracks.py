import torch
from simulation.env import PodRacerEnv
from config import *

def verify_tracks():
    print("Initializing Environment...")
    env = PodRacerEnv(num_envs=128, device='cuda')
    
    print("Testing Reset (Generation)...")
    env.reset()
    
    print("Checkpoints Shape:", env.checkpoints.shape)
    
    # Check bounds
    border = 1800.0
    cps = env.checkpoints
    
    in_bounds_x = (cps[:, :, 0] >= border) & (cps[:, :, 0] <= WIDTH - border)
    in_bounds_y = (cps[:, :, 1] >= border) & (cps[:, :, 1] <= HEIGHT - border)
    
    # Ignore 0,0 points (unused CPs)
    # Mask unused:
    # We need to mask based on n_cps
    
    valid_mask = torch.zeros_like(in_bounds_x)
    for i in range(128):
        n = env.num_checkpoints[i]
        valid_mask[i, :n] = True
    
    cps_active = cps[valid_mask]
    
    in_bounds = (cps_active[:, 0] >= border - 1.0) & (cps_active[:, 0] <= WIDTH - border + 1.0) & \
                (cps_active[:, 1] >= border - 1.0) & (cps_active[:, 1] <= HEIGHT - border + 1.0)
    
    if in_bounds.all():
        print("Bounds Check Passed!")
    else:
        print("Bounds Check FAILED!")
        print("First Fail:", cps_active[~in_bounds][0])

    print("Checking Intra-Checkpoint Distances (Overlaps)...")
    # Check min dist
    MIN_DIST_ALLOWED = 2500.0 
    
    # We need to check per env
    fail_overlap_count = 0
    
    for i in range(128):
        n = env.num_checkpoints[i]
        c = cps[i, :n] # [M, 2]
        
        # dist matrix
        # Expand
        p1 = c.unsqueeze(1)
        p2 = c.unsqueeze(0)
        dist = torch.norm(p1 - p2, dim=2)
        
        # Upper triangle only to ignore diagonal (0) and duplicates
        mask = torch.triu(torch.ones_like(dist), diagonal=1).bool()
        
        dists = dist[mask]
        
        if (dists < MIN_DIST_ALLOWED).any():
            fail_overlap_count += 1
            if fail_overlap_count <= 5:
                print(f"Env {i} Failure: Min Dist found {dists.min().item():.1f} < {MIN_DIST_ALLOWED}")
                
    if fail_overlap_count == 0:
        print("Overlap Check Passed!")
    else:
        print(f"Overlap Check FAILED! {fail_overlap_count}/128 Envs have clashing checkpoints.")

    print("Checking Generation Types...")
    # We can't easily query which Gen was used, but we can infer from geometry?
    # Not necessary. If it runs without crash, that's good.
    
    # Check Star Convexity? 
    # Just run multiple resets to ensure stability.
    for i in range(5):
        print(f"Reset {i+1}...")
        env.reset()
        
    print("Verification Complete.")

if __name__ == "__main__":
    verify_tracks()
