
import torch
from simulation.env import PodRacerEnv, STAGE_DUEL_FUSED

def debug_intercept():
    env = PodRacerEnv(num_envs=1, device='cpu', start_stage=STAGE_DUEL_FUSED)
    
    # Setup Scenario:
    # Me (Blocker) at (0,0), Velocity (0,0)
    # Enemy (Runner) at (1000, 1000), Velocity (1000, 0) (Moving East fast)
    # Next CP for Enemy is (1000, 5000) (North)
    
    # 1. Setup State
    env.physics.pos.fill_(0.0)
    env.physics.vel.fill_(0.0)
    
    me_idx = 1 # Blocker
    en_idx = 2 # Enemy Runner
    
    me_pos = torch.tensor([[1000.0, 3000.0]]) # Closer to CP (1000, 5000)
    en_pos = torch.tensor([[1000.0, 1000.0]])
    en_vel = torch.tensor([[1000.0, 0.0]]) # Moving away/laterally
    
    env.physics.pos[:, me_idx] = me_pos
    env.physics.pos[:, en_idx] = en_pos
    env.physics.vel[:, en_idx] = en_vel
    
    # Enemy Target CP
    env.next_cp_id[:, en_idx] = 0
    env.checkpoints[:, 0] = torch.tensor([1000.0, 5000.0])
    
    # Roles
    env.is_runner[:, me_idx] = False
    env.is_runner[:, en_idx] = True
    
    # 2. Call _get_front_intercept_pos
    # We need to simulate the batch call
    # This function expects [B, 2] inputs
    # [UPDATED] Pass me_vel for dynamic calculation
    me_vel = torch.zeros_like(me_pos) # Stationary blocker
    intercept_target = env._get_front_intercept_pos(me_pos, en_pos, en_vel, me_vel)
    
    print("\n--- INTERCEPT LOGIC ---")
    print(f"Me Pos: {me_pos[0].tolist()}")
    print(f"En Pos: {en_pos[0].tolist()}")
    print(f"En Vel: {en_vel[0].tolist()}")
    print(f"Me Vel: {me_vel[0].tolist()}")
    print(f"Intercept Target (Predicted): {intercept_target[0].tolist()}")
    
    # Verify Math manually
    dist = torch.norm(en_pos - me_pos)
    
    # Dynamic Math:
    # Rel Vel = En (1000, 0) - Me (0, 0) = (1000, 0)
    # Dir = En - Me = (1000, 1000) / sqrt(2M) = (0.707, 0.707)
    # Closing Speed = - (Rel . Dir) = - (1000*0.707 + 0) = -707.0 (Opening!)
    # Clamp(closing, 100, 2500) -> 100.0 (Since it's opening, we assume worst case slow approach?)
    # Wait, if they are moving AWAY, time to intercept should be large/infinite?
    # Logic: t_int = dist / speed_est.
    # speed_est = clamp(-707, 100, 2500) = 100.0.
    # t_int = dist / 100.0.
    
    # Dist = 1414.2
    # t_int = 14.14
    
    # Pred Pos = (1000, 1000) + (1000, 0) * 14.14 = (15140, 1000).
    # Clamped to 50 steps? t_int = clamp(..., 0, 50).
    # 14.14 is < 50. So 15140? That's far.
    
    # Previous Logic (Hardcoded 400):
    # t_int = 1414 / 400 = 3.535.
    # Pred Pos = (1000, 1000) + (1000, 0) * 3.535 = (4535, 1000).
    
    # Check what we get.

    
    # 3. Check Orientation Logic (Lines 870+)
    print("\n--- ORIENTATION LOGIC ---")
    
    # Logic from env.py:
    # d_me_cp vs d_en_cp
    target_cp_pos = env.checkpoints[:, 0]
    d_me_cp = torch.norm(target_cp_pos - me_pos)
    d_en_cp = torch.norm(target_cp_pos - en_pos)
    
    print(f"Dist Me->CP: {d_me_cp.item():.2f}")
    print(f"Dist En->CP: {d_en_cp.item():.2f}")
    
    is_ahead = (d_me_cp < d_en_cp)
    print(f"Is Ahead: {is_ahead.item()}")
    
    if is_ahead:
        final_target = en_pos
        print("Strategy: Target Enemy (Current Pos)")
    else:
        final_target = target_cp_pos
        print("Strategy: Target CP")
        
    print(f"Orientation Target: {final_target[0].tolist()}")
    
    # 4. Compare
    vec_intercept = intercept_target - me_pos
    vec_orient = final_target - me_pos
    
    angle_intercept = torch.atan2(vec_intercept[:, 1], vec_intercept[:, 0]).item() * 180 / 3.14159
    angle_orient = torch.atan2(vec_orient[:, 1], vec_orient[:, 0]).item() * 180 / 3.14159
    
    print(f"\nAngle to Intercept (Reward): {angle_intercept:.2f} deg")
    print(f"Angle to Orient (Guidance): {angle_orient:.2f} deg")
    print(f"Diff: {abs(angle_intercept - angle_orient):.2f} deg")
    
    if abs(angle_intercept - angle_orient) > 10.0:
        print("\n[CONCLUSION] INTERFERENCE CONFIRMED.")
    else:
        print("\n[CONCLUSION] No critical interference.")

if __name__ == "__main__":
    debug_intercept()
