
import torch
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from simulation.env import PodRacerEnv, STAGE_TEAM
from config import TrainingConfig, EnvConfig

def verify_bot_enhancements():
    print("--- Verifying Bot Enhancements ---")
    
    # 1. Setup Env (Team Stage for Boost)
    cfg = TrainingConfig()
    env_cfg = EnvConfig(active_pods=[0, 1, 2, 3], use_bots=True, bot_pods=[2, 3])
    env = PodRacerEnv(num_envs=1, device='cpu', start_stage=STAGE_TEAM)
    env.set_stage(STAGE_TEAM, env_cfg, reset_env=True)
    
    # Force Difficulty to Max
    env.bot_difficulty = 1.0
    print(f"Bot Difficulty: {env.bot_difficulty}")
    print(f"Stage: {env.curriculum_stage} (Team)")
    
    # 2. Scenario 1: Boost Verification (Long Range Catch-up)
    # Reset
    env.reset()
    
    # Configure Positions
    # Pod 2 is Bot-Runner (usually). Let's check roles.
    # We need a Bot BLOCKER.
    # In Team stage, roles are adaptive.
    # Let's force Pod 2 (Bot) to be Blocker (Runner=False) and Pod 0 (Agent) to be Runner.
    env.is_runner[:, 0] = True
    env.is_runner[:, 2] = False # Bot Blocker
    
    # Position:
    # Bot (2) at Origin
    env.physics.pos[:, 2, 0] = 0.0
    env.physics.pos[:, 2, 1] = 0.0
    env.physics.vel[:, 2] = 0.0
    env.physics.angle[:, 2] = 0.0 # Facing +X
    
    # Agent (0) Far away in front (Catch up scenario)
    env.physics.pos[:, 0, 0] = 5000.0
    env.physics.pos[:, 0, 1] = 0.0
    env.physics.vel[:, 0] = 100.0 # Moving away
    
    # Agent (0) needs to be target. 
    # Logic: Blocker targets Runner? Yes.
    
    # Step
    actions = torch.zeros((1, 4, 4))
    env.step(actions, None)
    
    # Check Boost
    # boosting is tracked in physics state? Or we can hack verify 'act_boost' inside step?
    # We can check physics velocity boost or fuel? 
    # Env doesn't expose 'act_boost' directly after step.
    # But if boost was used, speed should be higher than normal thrust?
    # Or fuel consumed?
    # 'physics.boost_available' might be false if cooldown trig?
    # Wait, 'act_boost' variable is local in step.
    # We can rely on 'physics.shield_cd' or similar if boost affects state.
    # Actually, let's just use print debugging in the script? No, I can't see prints easily.
    # I'll check if velocity is increasing rapidly or if boost state changes.
    # Currently physics model: Boost adds force.
    
    # Let's check internal flag if possible, or just trust the logic if it runs without error?
    # Better: The script should modify env to expose 'last_actions' or something?
    # Or I can just check if velocity > normal max?
    # Normal Thrust max acceleration ~ 20-100 force / mass.
    # Boost adds huge force.
    
    # Let's run for 10 steps and check position delta.
    
    print("\n[Scenario 1: Catch-up Boost]")
    start_x = env.physics.pos[0, 2, 0].item()
    for i in range(5):
        env.step(actions, None)
        vel = torch.norm(env.physics.vel[0, 2]).item()
        print(f"Step {i}: Bot Vel {vel:.1f}")
        
    final_x = env.physics.pos[0, 2, 0].item()
    dist = final_x - start_x
    print(f"Distance Covered in 5 steps: {dist:.1f}")
    
    if dist > 300.0: # Arbitrary threshold for boosted speed?
        print(">> HIGH SPEED DETECTED (Likely Boosting)")
    else:
        print(">> LOW SPEED (No Boost?)")

    # 3. Scenario 2: Active Intercept (Lateral Cut)
    print("\n[Scenario 2: Active Intercept]")
    # Bot (2) at Origin
    env.physics.pos[:, 2] = 0.0
    env.physics.vel[:, 2] = 0.0
    
    # Agent (0) Moving Laterally far away
    # CP at (10000, 0)
    cp_idx = env.next_cp_id[0, 0]
    cp_pos = env.checkpoints[0, cp_idx]
    print(f"Target CP Pos: {cp_pos}")
    
    # Place Agent at (10000, 5000) moving towards CP (10000, 0)
    # Actually let's put Agent at (5000, 5000) moving to (10000, 0)?
    # CP is just a point.
    # Let's make Agent cross the field.
    # Agent Pos: (5000, 5000). Vel: (500, -500) -> Towards (10000, 0) approx.
    env.physics.pos[:, 0, 0] = 5000.0
    env.physics.pos[:, 0, 1] = 5000.0
    env.physics.vel[:, 0, 0] = 500.0
    env.physics.vel[:, 0, 1] = -500.0
    
    # CP is likely (0,0) or random?
    # Let's force CP to (10000, 0) for clarity?
    # Can't easily force CP without modifying internal state.
    # We will just read CP.
    
    # Step
    env.step(actions, None)
    
    # Check Bot Direction (Angle)
    bot_angle = env.physics.angle[0, 2].item()
    print(f"Bot Angle: {bot_angle:.1f} deg")
    
    # Calculate Expected Gatekeeper Angle (Towards CP approx)
    # CP vector
    bot_pos = env.physics.pos[0, 2]
    vec_to_cp = cp_pos - bot_pos
    angle_to_cp = torch.rad2deg(torch.atan2(vec_to_cp[1], vec_to_cp[0])).item()
    print(f"Angle to CP (Gatekeeper direction): {angle_to_cp:.1f} deg")
    
    # Calculate Intercept Angle (Towards predicted intersection)
    # Agent is at (5000, 5000) moving South-East. 
    # Intercept should be somewhere between Bot and Agent?
    # If Bot is at (0,0), it should aim North-East to cut off?
    # Agent is at (5000, 5000).
    vec_to_agent = env.physics.pos[0, 0] - bot_pos
    angle_to_agent = torch.rad2deg(torch.atan2(vec_to_agent[1], vec_to_agent[0])).item()
    print(f"Angle to Agent (Direct): {angle_to_agent:.1f} deg")
    
    # Active Intercept should be aiming ahead of agent? or between CP and Agent?
    # Logic: intercept_pos = _get_front_intercept_pos
    # If diff=1.0, it targets intercept.
    
    diff = abs(bot_angle - angle_to_cp)
    print(f"Deviation from Gatekeeper (CP): {diff:.1f} deg")
    
    if diff > 10.0:
        print(">> BOT IS NOT GATEKEEPING (Good, likely Intercepting)")
    else:
        print(">> BOT IS GATEKEEPING (Passive, Bad?)")

if __name__ == "__main__":
    verify_bot_enhancements()
