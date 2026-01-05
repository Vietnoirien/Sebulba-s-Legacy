
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.env import PodRacerEnv, STAGE_DUEL
from config import EnvConfig

def test_ghost_injection():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device}")
    
    # Initialize Env in Stage 2 (Duel)
    # Active Pods: [0, 2]. Inactive: [1, 3]
    env = PodRacerEnv(num_envs=10, device=device)
    config = EnvConfig(
        mode_name="duel",
        track_gen_type="max_entropy",
        active_pods=[0, 2],
        use_bots=True,
        bot_pods=[2],
        step_penalty_active_pods=[0, 2],
        orientation_active_pods=[0, 2]
    )
    env.set_stage(STAGE_DUEL, config, reset_env=True)
    
    # Check Physics State
    # Pod 1 is inactive, should be at -100,000
    p1_pos = env.physics.pos[:, 1]
    print(f"Physics Pod 1 Mean Pos: {p1_pos.mean(dim=0)}")
    
    if p1_pos.mean().item() > -90000:
        print("FAIL: Physics Pod 1 is not at Infinity!")
        return
        
    print("PASS: Physics Pod 1 is correctly at Infinity.")
    
    # Check Observations
    s, tm, en, cp = env.get_obs()
    
    # tm is [4, B, 13]
    # We want Teammate Obs for Pod 0 (which is Pod 1's features)
    # tm[0] -> Obs for Pod 0's teammate.
    tm0 = tm[0] # [B, 13]
    
    # Features: dp(2), dv(2), cos(1), sin(1), dist(1), ...
    # dist is index 6 (after dp(2)+dv(2)+cos(1)+sin(1))?
    # Let's check env.pycat ordering: dp_local, dv_local, rel_cos, rel_sin, dist, ...
    # 0,1 | 2,3 | 4 | 5 | 6 ...
    
    dist_vals = tm0[:, 6]
    
    print(f"Teammate Dist Vals (First 5): {dist_vals[:5]}")
    
    # Normalized dist. S_POS = 1/16000.
    # Ghost Range +/- 4000 relative + 3000 offset logic?
    # Logic was: P + Random(-4000, 4000).
    # So dist should be < 4000 * S_POS + noise
    # 4000 / 16000 = 0.25.
    
    # If it was Infinity, dist would be huge (or clipped/weird).
    # -100,000 - 0 = 100,000. / 16000 = 6.25.
    
    if dist_vals.mean() > 1.0:
        print(f"FAIL: Teammate Dist is too large ({dist_vals.mean()}). Ghost Injection failed?")
    else:
        print(f"PASS: Teammate Dist is reasonable ({dist_vals.mean()}). Ghost Injection ACTIVE.")
        
    if dist_vals.min() < 0.0:
        print("FAIL: Dist must be positive.")
        
    print("Verification Complete.")

if __name__ == "__main__":
    test_ghost_injection()
