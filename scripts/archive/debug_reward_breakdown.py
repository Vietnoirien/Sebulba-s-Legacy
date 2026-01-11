
import torch
import numpy as np
from simulation.env import PodRacerEnv, STAGE_DUEL_FUSED, DEFAULT_REWARD_WEIGHTS
from config import TrainingConfig, CurriculumConfig, EnvConfig
from training.curriculum.stages import UnifiedDuelStage

def debug_rewards():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainingConfig(device=device)
    
    # Test Cases
    scenarios = [
        {'name': "Far Stationary", 'dist': 4000.0, 'angle': 0.0, 'speed': 0.0},
        {'name': "Far Facing Enemy", 'dist': 4000.0, 'angle': 0.0, 'speed': 0.0, 'face_en': True},
        {'name': "Far Facing Away", 'dist': 4000.0, 'angle': 180.0, 'speed': 0.0},
        {'name': "Far Intercept Course", 'dist': 4000.0, 'angle': 0.0, 'speed': 800.0, 'intercept': True},
        {'name': "Mid Distance Intercept", 'dist': 2500.0, 'angle': 0.0, 'speed': 600.0, 'intercept': True},
        {'name': "Close Non-Contact", 'dist': 850.0, 'angle': 0.0, 'speed': 400.0, 'side': True}, # Side by side?
        
        # --- NEW: Collision & Orientation Tests ---
        # 1. Head-On (High Speed)
        {'name': "Head-On High Spd", 'dist': 600.0, 'custom': True, 
         'me_pos': [-300, 0], 'en_pos': [300, 0], 'me_vel': [800, 0], 'en_vel': [-800, 0]},
        
        # 2. Head-On (Low Speed)
        {'name': "Head-On Low Spd", 'dist': 600.0, 'custom': True,
         'me_pos': [-300, 0], 'en_pos': [300, 0], 'me_vel': [200, 0], 'en_vel': [-200, 0]},
         
        # 3. T-Bone (Me hitting En Side)
        {'name': "T-Bone Collision", 'dist': 600.0, 'custom': True,
         'me_pos': [0, -300], 'en_pos': [0, 300], 'me_vel': [0, 1000], 'en_vel': [400, 0]},
         
        # 4. Rear-End (High Delta)
        {'name': "Rear-End High Delta", 'dist': 600.0, 'custom': True,
         'me_pos': [-600, 0], 'en_pos': [0, 0], 'me_vel': [1000, 0], 'en_vel': [500, 0]},
         
        # 5. Rear-End (Low Delta)
        {'name': "Rear-End Low Delta", 'dist': 600.0, 'custom': True,
         'me_pos': [-600, 0], 'en_pos': [0, 0], 'me_vel': [550, 0], 'en_vel': [500, 0]},
         
        # 6. Pinning (Touching, pushing against En)
        {'name': "Pinning Contact", 'dist': 550.0, 'custom': True,
         'me_pos': [-275, 0], 'en_pos': [275, 0], 'me_vel': [100, 0], 'en_vel': [0, 0]},
         
        # --- Orientation Tests ---
        # Intercept Course (Head On)
        # Me: West of En (-1500). Moving East (+600).
        # En: At 0. Moving West (-400).
        # Dist decrease rate: 1000 u/s.
        
        {'name': "Intercept Face En", 'custom': True, 'dist': 1500.0,
         'me_pos': [-1500, 0], 'en_pos': [0, 0], 'me_vel': [600, 0], 'en_vel': [-400, 0],
         'angle_deg': 0.0}, # Facing East (Towards Enemy)
         
        {'name': "Intercept Face Away", 'custom': True, 'dist': 1500.0,
         'me_pos': [-1500, 0], 'en_pos': [0, 0], 'me_vel': [600, 0], 'en_vel': [-400, 0],
         'angle_deg': 180.0}, # Facing West (Away from Enemy)
        {'name': "Head-On Face Away", 'custom': True, 'dist': 600.0, 
         'me_pos': [-300, 0], 'en_pos': [300, 0], 'me_vel': [800, 0], 'en_vel': [-800, 0],
         'angle_deg': 180.0},
         
        {'name': "Head-On Low Spd Face Away", 'custom': True, 'dist': 600.0, 
         'me_pos': [-300, 0], 'en_pos': [300, 0], 'me_vel': [200, 0], 'en_vel': [-200, 0],
         'angle_deg': 180.0},
    ]
    
    num_scenarios = len(scenarios)
    print(f"DEBUG: Running {num_scenarios} scenarios...")

    env = PodRacerEnv(num_scenarios, device=device)
    
    # Init Env
    curr_config = CurriculumConfig()
    stage = UnifiedDuelStage(curr_config)
    env.set_stage(STAGE_DUEL_FUSED, stage.get_env_config())
    env.reset()
    
    # Configure Pods
    # Pod 0: Me (Blocker). Pod 2: Enemy (Runner).
    # Mask "is_runner".
    
    # Override
    env.is_runner[:] = False
    env.is_runner[:, 2] = True # Pod 2 is Runner
    
    # Setup Scenarios
    for i, scen in enumerate(scenarios):
        if scen.get('custom'):
            # Custom Setup (High/Low Speed Collisions)
            mp = torch.tensor(scen['me_pos'], device=device).float()
            ep = torch.tensor(scen['en_pos'], device=device).float()
            mv = torch.tensor(scen['me_vel'], device=device).float()
            ev = torch.tensor(scen['en_vel'], device=device).float()
            
            env.physics.pos[i, 0] = mp
            env.physics.pos[i, 2] = ep
            env.physics.vel[i, 0] = mv
            env.physics.vel[i, 2] = ev
            
            # Set Angle from Velocity or Override
            # Me
            if 'angle_deg' in scen:
                 rad = torch.tensor(scen['angle_deg'] * 3.14159 / 180.0, device=device)
                 env.physics.angle[i, 0] = rad
            elif torch.norm(mv) > 0.1:
                env.physics.angle[i, 0] = torch.atan2(mv[1], mv[0])
            
            # Enemy
            if torch.norm(ev) > 0.1:
                env.physics.angle[i, 2] = torch.atan2(ev[1], ev[0])
                
        else:
            # Standard Setup
            dist = scen['dist']
            angle_deg = scen['angle']
            speed = scen['speed']
            
            # Enemy at (0,0) basically
            env.physics.pos[i, 2] = torch.tensor([0.0, 0.0], device=device)
            env.physics.vel[i, 2] = torch.tensor([400.0, 0.0], device=device) # Moving East
            env.physics.angle[i, 2] = torch.tensor(0.0, device=device)
            
            # Me
            rad = torch.tensor(angle_deg * 3.14159 / 180.0, device=device)
            mx = dist * torch.cos(rad)
            my = dist * torch.sin(rad)
            
            if scen.get('side'):
                 # Side Test (e.g. 850u lateral)
                 env.physics.pos[i, 0] = torch.tensor([0.0, -dist], device=device)
            else:
                 env.physics.pos[i, 0] = torch.tensor([mx, my], device=device)
                 
            # Orientation
            if scen.get('face_en'):
                 # Face towards 0,0
                 env.physics.angle[i, 0] = torch.atan2(-my, -mx)
            elif scen.get('intercept'):
                 # Velocity Vector to Enemy
                 d = torch.tensor([-mx, -my], device=device)
                 d = d / (torch.norm(d) + 1e-6)
                 env.physics.vel[i, 0] = d * speed
                 env.physics.angle[i, 0] = torch.atan2(d[1], d[0])
            else:
                 env.physics.angle[i, 0] = torch.tensor(0.0, device=device)
                 env.physics.vel[i, 0] = torch.zeros(2, device=device) 
    # Scenario Setup Complete.
    # Checkpoints setup
    env.next_cp_id[:, 2] = 0
    env.checkpoints[:, 0] = torch.tensor([12000.0, 4500.0], device=device) # Goal East

             
    
    # Init Weights
    w_tensor = torch.zeros((num_scenarios, 20), device=device)
    for k, v in DEFAULT_REWARD_WEIGHTS.items():
        if k < 20:
             w_tensor[:, k] = v
             
    # Run Simulation Loop
    # We need to run enough steps to settle "Delta" rewards
    print(f"{'Scenario':<25} | {'Dist':<6} | {'Reward (Last)':<15} | {'Blocker Dmg (Last)':<18} | {'Avg Rew (Last 5)':<15}")
    print("-" * 90)

    # We cannot easily vectorized-set positions every step for a static test because physics moves them.
    # But for "Far Stationary", they stay (approx).
    # For "Intercept", they move.
    
    # We want to measure the reward AT that state.
    # To do this without drift, we must reset the state every step?
    # No, that triggers the Delta bug.
    
    # Solution: Manually fix "Prev" variables to match current "Manual" state before the first step.
    # Then run 1 step.
    
    # 1. Update Physics (Already done)
    
    # 2. Update Prevs
    # Update prev_dist based on manual positions
    env.update_progress_metric(torch.arange(num_scenarios, device=device))
    
    # Update prev_dist_intercept
    # We need to call _get_front... manually?
    me_pos = env.physics.pos[:, 0]
    en_pos = env.physics.pos[:, 2] # Real enemy
    en_vel = env.physics.vel[:, 2]
    me_vel = env.physics.vel[:, 0]
    
    # Recalculate intercept pos to set prev
    with torch.no_grad():
        int_pos = env._get_front_intercept_pos(me_pos, en_pos, en_vel, me_vel)
        dist_int = torch.norm(int_pos - me_pos, dim=1)
        env.prev_dist_intercept[:, 0] = dist_int
        
    # Now run 1 step to get the "Instantaneous" reward without Delta Artifacts
    rewards, _, infos = env.step(torch.zeros((num_scenarios, 4, 4), device=device), w_tensor)
    
    bdm = infos.get('blocker_damage', torch.zeros_like(rewards))
    dzm = infos.get('denial_zone', torch.zeros_like(rewards))
    
    for i, scen in enumerate(scenarios):
        r = rewards[i, 0].item()
        bd = bdm[i, 0].item()
        dz = dzm[i, 0].item()
        
        print(f"{scen['name']:<25} | {scen['dist']:<6.0f} | {r:<15.2f} | {bd:<18.2f} | {dz:<15.2f}")


if __name__ == "__main__":
    debug_rewards()
