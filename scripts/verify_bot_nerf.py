
import torch
import torch.nn as nn
from simulation.env import PodRacerEnv, STAGE_DUEL

def verify_bot_difficulty():
    print("Initializing Environment...")
    env = PodRacerEnv(num_envs=128, device='cuda', start_stage=STAGE_DUEL)
    
    # --- Test Difficulty 0.0 (The Nerf) ---
    print("\n--- Testing Difficulty 0.0 ---")
    env.bot_difficulty = 0.0
    env.reset()
    
    # We need to peek into the step function or inspect physics after step
    # Actions are local to step, but we can verify physics velocity behavior or 
    # mock the step to capture actions?
    # Actually, env.step calculates actions internally for the bot.
    # We can't easily see the action tensor unless we instrument env.
    # BUT, we can check the velocity magnitude after 1 step from rest.
    # Thrust = 40.0. Mass = 1.0. Friction = 0.85.
    # Impulse = 40.0 * 100? No, step applies thrust directly? 
    # physics.step: 
    # acc = thrust * orientation
    # vel += acc * dt
    # let's assume standard physics.
    
    # Creating a temporary wrapper or just monkey-patching step might be complex.
    # Alternative: We can recalculate what the action WOULD be using the same logic, 
    # since we have access to env state.
    
    # Bot Logic from env.py:
    # thrust_val = 40.0 + (60.0 * diff)
    
    expected_thrust_0 = 40.0
    expected_thrust_1 = 100.0
    
    # Let's verify by just calculating the formula directly as a sanity check of the code we just wrote?
    # No, we want to run the code.
    
    # Run 10 steps and check average speed?
    # Max speed is limited by friction.
    # Terminal velocity ~ Thrust / (1-Friction)? Or Thrust / Friction_coeff?
    # simulation is usually: vel = vel * friction + thrust
    # Terminal = Thrust / (1 - 0.85) = Thrust / 0.15 = Thrust * 6.66
    # 40 * 6.66 = 266
    # 60 * 6.66 = 400
    # 100 * 6.66 = 666
    
    # So we should see significantly lower speeds for bot.
    
    env.bot_difficulty = 0.0
    env.reset()
    
    # Let's run for 50 steps
    # Bot is index 2
    for _ in range(50):
        actions = torch.zeros((128, 4, 4), device='cuda')
        env.step(actions)
        
    vel = env.physics.vel[:, 2].norm(dim=1)
    mean_vel_0 = vel.mean().item()
    print(f"Diff 0.0 Mean Velocity (50 steps): {mean_vel_0:.2f}")
    
    # --- Test Difficulty 1.0 (Max) ---
    print("\n--- Testing Difficulty 1.0 ---")
    env.bot_difficulty = 1.0
    env.reset()
    
    for _ in range(50):
        actions = torch.zeros((128, 4, 4), device='cuda')
        env.step(actions)
        
    vel = env.physics.vel[:, 2].norm(dim=1)
    mean_vel_1 = vel.mean().item()
    print(f"Diff 1.0 Mean Velocity (50 steps): {mean_vel_1:.2f}")
    
    if mean_vel_0 < mean_vel_1 * 0.6: # 266 vs 666 is ~40%
        print("\n[PASS] Difficulty 0.0 is significantly slower than Difficulty 1.0")
    else:
        print("\n[FAIL] Velocities are too similar!")
        
    if mean_vel_0 < 300.0:
         print(f"[PASS] Diff 0.0 Velocity {mean_vel_0:.2f} is consistent with ~40% thrust.")
    else:
         print(f"[FAIL] Diff 0.0 Velocity {mean_vel_0:.2f} seems too high for 40% thrust.")

if __name__ == "__main__":
    verify_bot_difficulty()
