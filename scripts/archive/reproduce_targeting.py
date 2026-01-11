
import torch
from simulation.env import PodRacerEnv
from config import EnvConfig, STAGE_TEAM

def reproduction():
    print("Initializing Environment...")
    env = PodRacerEnv(num_envs=1, device='cpu')
    
    # Configure for Team Stage 2v2 with Bots
    config = EnvConfig(
        mode_name="team",
        use_bots=True, 
        bot_pods=[1, 3] # P1 and P3 are bots
    )
    
    # Disable Noise
    env = PodRacerEnv(num_envs=1, device='cpu')
    env.bot_config.difficulty_noise_scale = 0.0
    
    env.set_stage(STAGE_TEAM, config, reset_env=True)
    
    # Manually Override States
    # Team 0: P0 (Runner), P1 (Blocker - Bot)
    # Team 1: P2 (Runner), P3 (Blocker - Bot)
    
    # Scenario:
    # P1 (Blocker 0) is at Origin (0,0).
    # P0 (Runner 0 - Teammate) is at (0, 5000). Angle +90.
    # P2 (Runner 1 - Opponent) is at (5000, 0). Angle 0.
    
    # P1 should target P2 (Opponent).
    # If targeting P0 (Teammate), it's a BUG.
    
    # Set Positions
    env.physics.pos.fill_(0.0)
    env.physics.vel.fill_(0.0)
    env.physics.angle.fill_(0.0)
    
    # P1 Setup
    env.physics.pos[:, 1] = torch.tensor([0.0, 0.0])
    env.physics.angle[:, 1] = 0.0 # Facing +X (Towards P2)
    
    # P0 Setup (Teammate)
    env.physics.pos[:, 0] = torch.tensor([0.0, 5000.0])
    
    # P2 Setup (Opponent)
    env.physics.pos[:, 2] = torch.tensor([5000.0, 0.0])
    
    # Reset Checkpoints/NextCP to ensure valid targets
    # P0 (North Runner) Next CP: (0, 10000) - Pure North
    # P2 (East Runner) Next CP: (10000, 0) - Pure East
    
    # Checkpoints: [B, MaxCP, 2]
    # Set CP 0 for everyone just to be safe (Index 0)
    # But usually we look at next_cp_id.
    
    # We need to set CP positions specifically for the Runners' NextCP indices.
    # Let's say NextCP is 0 for everyone.
    env.next_cp_id.fill_(0)
    
    # BUT Checkpoints is SHARED across batch? No, [B, MaxCP, 2].
    # But P0 and P2 are in the SAME environment (Batch 0).
    # They share the SAME map and checkpoints.
    # This is the Catch! In a shared map, all checkpoints are same.
    # Run loop logic: P0 is Runner, P2 is Runner.
    # P0 going to CP0? P2 going to CP0?
    # In a race, they go to same CPs.
    
    # So if P0 is at (0, 5000) going to (0, 10000). CP must be (0, 10000).
    # Then P2 at (5000, 0) is ALSO going to (0, 10000).
    # P2 Vector: (5000, 0) -> (0, 10000) = (-5000, 10000). Angle ~116 deg.
    # P0 Vector: (0, 5000) -> (0, 10000) = (0, 5000). Angle 90 deg.
    
    # If P1 targets P0 (Intercepting P0 path): Target approx (0, 8000). Angle 90.
    # If P1 targets P2 (Intercepting P2 path): Target approx (-2500, 8000)? 
    # Intercept point for P2 is on the line P2->CP. 
    # Midpoint? Gatekeeper offset 2500 from CP.
    # CP (0, 10000).
    # P2 Dir: (-0.44, 0.89).
    # Intercept = CP - 2500*Dir = (0, 10000) - (-1100, 2200) = (1100, 7800).
    # Angle to (1100, 7800) from (0,0) ~ 82 degrees.
    # P1 (Angle 0). Delta 82. Output 18.
    
    # Still ambiguous both positive.
    
    # FIX: Use DIFFERENT NextCP indices for different pods?
    # env.next_cp_id is [B, 4]. We can assign different targets!
    # P0 target CP 0 (North).
    # P2 target CP 1 (East).
    
    env.next_cp_id[0, 0] = 0
    env.next_cp_id[0, 2] = 1
    
    # Set CP locations
    env.checkpoints[0, 0] = torch.tensor([0.0, 10000.0]) # North
    env.checkpoints[0, 1] = torch.tensor([10000.0, 0.0]) # East
    
    # Step Capture
    # We need to capture the 'act_angle' inside the step function.
    # Since we can't easily hook, we will check the result of physics.
    # With physics step, if steering is applied, angle changes.
    # Angle change = act_angle
    # act_angle = clamp(delta, -18, 18).
    
    # If P1 targets P0 (Angle 90):
    # Desired = 90. Current = 0. Delta = 90. Clamped = 18.
    # Physics performs integration. Angle += 18 (roughly).
    
    # If P1 targets P2 (Angle 0):
    # Desired = 0. Current = 0. Delta = 0.
    # Angle += 0.
    
    print("\n--- TEST CASE A: P1 (Bot Blocker) ---")
    print("P1 Pos: (0,0), Facing East (0 deg)")
    print("P0 (Teammate) Pos: (0,5000) North (90 deg)")
    print("P2 (Opponent) Pos: (5000,0) East (0 deg)")
    print("Expected: P1 targets P2 (Steering ~ 0)")
    print("Bug: P1 targets P0 (Steering > 0)")
    
    # Dummy Actions (All Zeros)
    dummy_actions = torch.zeros((1, 4, 4))
    
    # Run Step
    env.step(dummy_actions, reward_weights=None)
    
    # Check Result
    new_angle = env.physics.angle[0, 1].item()
    print(f"P1 Angle After Step: {new_angle:.2f}")
    
    if new_angle > 5.0:
        print("FAIL: P1 turned Left (Towards Teammate P0). BUG CONFIRMED.")
    elif abs(new_angle) < 5.0:
        print("PASS: P1 stayed straight (Towards Opponent P2).")
    else:
        print(f"UNCLEAR: {new_angle}")

    # --- TEST CASE B: P3 (Opponent Bot Blocker) ---
    # P3 at (0,0). P0 (Opponent) at (0, 5000). P2 (Teammate) at (5000, 0).
    # P3 Facing P0 (90 deg).
    # Should stay 90 (Target Opponent P0).
    # If targeting Teammate P2 (0 deg), should turn Right (-).
    
    # Reset
    env.physics.pos.fill_(0.0)
    env.physics.vel.fill_(0.0)
    
    env.physics.pos[:, 3] = 0.0
    env.physics.angle[:, 3] = 90.0 # Facing North (Towards P0)
    env.physics.pos[:, 0] = torch.tensor([0.0, 5000.0]) # Opponent
    env.physics.pos[:, 2] = torch.tensor([5000.0, 0.0]) # Teammate
    
    print("\n--- TEST CASE B: P3 (Bot Blocker) ---")
    print("P3 Pos: (0,0), Facing North (90 deg)")
    print("P0 (Opponent) Pos: (0,5000) North (90 deg)")
    print("P2 (Teammate) Pos: (5000,0) East (0 deg)")
    print("Expected: P3 targets P0 (Steering ~ 0)")
    print("Bug: P3 targets P2 (Steering < 0)")
    
    env.step(dummy_actions, reward_weights=None)
    
    new_angle_3 = env.physics.angle[0, 3].item()
    print(f"P3 Angle After Step: {new_angle_3:.2f}")
    
    if abs(new_angle_3 - 90.0) < 5.0:
        print("PASS: P3 stayed straight (Towards Opponent P0).")
    else:
        print("FAIL: P3 turned away from Opponent.")

if __name__ == "__main__":
    reproduction()
