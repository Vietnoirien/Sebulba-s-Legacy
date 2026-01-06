
import torch
import os
import sys
import numpy as np

# Add Project Root
sys.path.append(os.path.abspath('.'))

from simulation.env import PodRacerEnv, STAGE_DUEL, STAGE_TEAM
from config import TrainingConfig
from models.deepsets import PodAgent
from training.normalization import RunningMeanStd

def debug_transition():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEBUG: Using device {device}")

    # 1. Load Gen 52 (Last Stage 2)
    # We assume the user has this gen. If not, we might need to find the latest available.
    gen_dir = "data/generations/gen_51"
    agent_path = os.path.join(gen_dir, "agent_0.pt")
    
    if not os.path.exists(agent_path):
        print(f"ERROR: Could not find {agent_path}. Please adjust script to point to a valid Stage 2 checkpoint.")
        # Try to find any gen
        gens = [d for d in os.listdir("data/generations") if d.startswith("gen_")]
        if not gens:
            print("No generations found.")
            return
        latest = sorted(gens, key=lambda x: int(x.split('_')[1]))[-1]
        gen_dir = os.path.join("data/generations", latest)
        agent_path = os.path.join(gen_dir, "agent_0.pt")
        print(f"Fallback: Using {agent_path}")

    print(f"Loading Agent from {agent_path}...")
    agent = PodAgent()
    state_dict = torch.load(agent_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.to(device)
    agent.eval()
    
    # Load RMS Stats
    rms_path = os.path.join(gen_dir, "rms_stats.pt")
    rms_self = RunningMeanStd(shape=(15,), device=device)
    rms_ent = RunningMeanStd(shape=(13,), device=device)
    rms_cp = RunningMeanStd(shape=(6,), device=device)
    
    if os.path.exists(rms_path):
        print("Loading RMS Stats...")
        rms_data = torch.load(rms_path, map_location=device)
        rms_self.load_state_dict(rms_data['self'])
        rms_ent.load_state_dict(rms_data['ent'])
        rms_cp.load_state_dict(rms_data['cp'])
    else:
        print("WARNING: No RMS stats found. Using Identity.")

    # 2. Setup Environments
    print("\n--- STAGE 2 (DUEL) CONTROL ---")
    env_duel = PodRacerEnv(num_envs=1, device=device, start_stage=STAGE_DUEL)
    env_duel.reset()
    
    obs_duel = env_duel.get_obs()
    # obs is tuple (self, tm, en, cp) of shape [4, num_envs, ...]
    # We want Pod 0
    raw_s = obs_duel[0][0] # [1, 15]
    raw_tm = obs_duel[1][0] # [1, 13]
    raw_en = obs_duel[2][0] # [1, 2, 13]
    raw_cp = obs_duel[3][0] # [1, 6]
    
    # Normalize
    n_s = rms_self(raw_s)
    n_tm = rms_ent(raw_tm)
    # Enemy: [1, 2, 13] -> view(-1, 13) -> norm -> view
    n_en = rms_ent(raw_en.view(-1, 13)).view(1, 2, 13) 
    n_cp = rms_cp(raw_cp)
    
    # Inference Stage 2
    with torch.no_grad():
        act_d, _, _, _ = agent.get_action_and_value(n_s, n_tm, n_en, n_cp)
    
    print(f"Duel Action (Pod 0): {act_d[0, 0].cpu().numpy()}")
    print(f"Duel Runner Prob: {agent.self_obs[0, 11].item() if hasattr(agent, 'self_obs') else 'N/A'}")

    
    print("\n--- STAGE 3 (TEAM) EXPERIMENT ---")
    env_team = PodRacerEnv(num_envs=1, device=device, start_stage=STAGE_TEAM)
    env_team.reset()
    
    obs_team = env_team.get_obs()
    
    # Normalize (Pod 0)
    raw_s_t = obs_team[0][0]
    raw_tm_t = obs_team[1][0]
    raw_en_t = obs_team[2][0]
    raw_cp_t = obs_team[3][0]

    t_s = rms_self(raw_s_t)
    t_tm = rms_ent(raw_tm_t) # Now a REAL teammate?
    t_en = rms_ent(raw_en_t.view(-1, 13)).view(1, 2, 13)
    t_cp = rms_cp(raw_cp_t)
    
    # Inference Stage 3
    with torch.no_grad():
         act_t, _, _, _ = agent.get_action_and_value(t_s, t_tm, t_en, t_cp)
         
    print(f"Team Action (Pod 0): {act_t[0, 0].cpu().numpy()}")
    
    # 3. MITOSIS SIMULATION
    print("\n--- SIMULATING MITOSIS FIX ---")
    # Clone Runner -> Blocker
    agent.blocker_actor.load_state_dict(agent.runner_actor.state_dict())
    
    # Zero Teammate Input (The Fix)
    with torch.no_grad():
        agent.runner_actor.commander.backbone[0].weight[:, 15:31].zero_()
        agent.blocker_actor.commander.backbone[0].weight[:, 15:31].zero_()
    
    # Inference Post-Mitosis
    with torch.no_grad():
        act_m, _, _, _ = agent.get_action_and_value(t_s, t_tm, t_en, t_cp)
        
    print(f"Mitosis Action (Pod 0): {act_m[0, 0].cpu().numpy()}")
    
    # Compare
    print("\n--- COMPARISON ---")
    print(f"Duel Thrust: {act_d[0, 0].item():.4f}")
    print(f"Team Thrust: {act_t[0, 0].item():.4f}")
    print(f"Mito Thrust: {act_m[0, 0].item():.4f}")
    
    if abs(act_d[0,0].item() - act_m[0,0].item()) > 0.1:
        print("WARNING: Significant drift even after Mitosis fix!")
    else:
        print("SUCCESS: Mitosis fix results in stable initial action.")

if __name__ == "__main__":
    debug_transition()
