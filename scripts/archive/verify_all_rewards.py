import torch
import numpy as np
from simulation.env import PodRacerEnv
from config import TrainingConfig, STAGE_DUEL_FUSED, STAGE_SOLO

def verify_all_rewards():
    print("=== FULL REWARD VERIFICATION ===")
    
    cfg = TrainingConfig()
    cfg.num_envs = 4
    cfg.device = 'cpu'
    
    env = PodRacerEnv(4, device='cpu')
    # Force Stage 2 to enable all rewards
    env.curriculum_stage = STAGE_DUEL_FUSED 
    env.reset()
    
    # --- SETUP ROLES ---
    # Helper to teleport and update internal state
    def teleport(pod_idx, pos_tensor):
        env.physics.pos[0, pod_idx] = pos_tensor
        # Update prev_dist to avoid huge progress rewards/penalties
        # Need to know target... assuming next_cp_id=1 for this test
        target = env.checkpoints[0, env.next_cp_id[0, pod_idx]]
        dist = torch.norm(target - pos_tensor)
        env.prev_dist[0, pod_idx] = dist

    
    # Default Weights
    # We create a weights tensor that matches standard config
    # We'll override specific weights for strict testing if needed, but better to test "as is"
    weights = torch.ones((4, 25), device='cpu')
    # Populate specific indices based on config.py constants
    # w = env.config_rewards # Removed
    from config import DEFAULT_REWARD_WEIGHTS
    for k, v in DEFAULT_REWARD_WEIGHTS.items():
        weights[:, k] = v
        
    print(f"Weights Loaded. Win={weights[0,0]}, CP={weights[0,2]}, Denial={weights[0,15]}")
    
    # =========================================================================
    # TEST 1: RUNNER CHECKPOINT & GOALIE PENALTY
    # =========================================================================
    print("\n--- Test 1: Runner Checkpoint & Goalie Penalty ---")
    env.reset()
    # P0 (Runner) just before CP1
    cp1_pos = env.checkpoints[0, 1]
    
    # Teleport P0 to CP1 (Trigger Pass)
    teleport(0, cp1_pos)
    
    # Teleport others away
    teleport(3, torch.tensor([0.0, 0.0]))
    teleport(1, torch.tensor([-2000.0, -2000.0]))
    teleport(2, torch.tensor([-2000.0, -2000.0]))
    
    # Step
    ret = env.step(torch.zeros(4, 4, 4), weights)
    if len(ret) == 3: rew, _, _ = ret
    else: _, rew, _, _ = ret
    
    r_run = rew[0, 0].item()
    r_block_partner = rew[0, 1].item()
    r_run_opp = rew[0, 2].item()
    r_block_opp = rew[0, 3].item()
    
    print(f"Runner (P0) Reward: {r_run:.2f}")
    print(f"Partner Blocker (P1) Reward: {r_block_partner:.2f}")
    print(f"Opponent Runner (P2) Reward: {r_run_opp:.2f}")
    print(f"Opponent Blocker (P3) Reward: {r_block_opp:.2f}")
    
    if r_block_opp > 0:
        print("!!! CRITICAL FAILURE: Blocker GAINED reward from opponent scoring !!!")
    else:
        print("Goalie Penalty Logic: OK (Opponent Blocker received negative)")

    # =========================================================================
    # TEST 1B: RANK ISOLATION (Verify P2 Penalty Source)
    # =========================================================================
    print("\n--- Test 1B: Rank Isolation (RW_RANK=0) ---")
    env.reset()
    # Isolate Rank
    weights_no_rank = weights.clone()
    weights_no_rank[:, 13] = 0.0 # RW_RANK (13)
    weights_no_rank[:, 4] = 0.0  # RW_PROGRESS (4)
    weights_no_rank[:, 8] = 0.0  # RW_ORIENTATION (8)
    weights_no_rank[:, 11] = 0.0 # RW_PROXIMITY (11)
    
    # Teleport Same as Test 1
    cp1_pos = env.checkpoints[0, 1]
    teleport(0, cp1_pos)
    teleport(3, torch.tensor([0.0, 0.0]))
    teleport(1, torch.tensor([-2000.0, -2000.0]))
    teleport(2, torch.tensor([-2000.0, -2000.0]))
    
    # Enable correct roles manually (P2 is Runner)
    env.is_runner[:] = False
    env.is_runner[0, 0] = True
    env.is_runner[0, 2] = True
    
    ret = env.step(torch.zeros(4, 4, 4), weights_no_rank)
    if len(ret) == 3: rew, _, _ = ret
    else: _, rew, _, _ = ret
    
    r_run_opp = rew[0, 2].item()
    r_block_opp = rew[0, 3].item()
    
    print(f"Opponent Runner (P2) Reward (No Rank): {r_run_opp:.2f} (Expected 0.0)")
    print(f"Opponent Blocker (P3) Reward (No Rank): {r_block_opp:.2f} (Expected -200.0) (Opponent Blocker received negative)")

    # =========================================================================
    # TEST 2: VELOCITY DENIAL
    # =========================================================================
    print("\n--- Test 2: Velocity Denial ---")
    env.reset()
    # P1 (Blocker) vs P2 (Enemy Runner)
    # Enable Denial Weight
    env.reward_scaling_config.velocity_denial_weight = 100.0
    env.reward_scaling_config.intercept_progress_scale = 0.0 # Isolate
    
    # P2 (Enemy) racing to CP
    cp_next = env.checkpoints[0, 1]
    env.next_cp_id[0, 2] = 1
    
    # Teleport P2 to Origin
    teleport(2, torch.tensor([0.0, 0.0]))
    
    v_dir = (cp_next - torch.tensor([0.0, 0.0]))
    v_dir = v_dir / torch.norm(v_dir)
    env.physics.vel[0, 2] = v_dir * 800.0 # Max Speed
    
    # P1 (Me) stationary
    teleport(1, torch.tensor([100.0, 100.0]))
    env.physics.vel[0, 1] = torch.zeros(2)
    
    ret = env.step(torch.zeros(4, 4, 4), weights)
    if len(ret) == 3: rew, _, _ = ret
    else: _, rew, _, _ = ret
    
    r_denial = rew[0, 1].item()
    print(f"Blocker Denial Reward (Enemy Racing): {r_denial:.2f} (Expected ~ -85)")
    
    # =========================================================================
    # TEST 3: COLLISION REWARDS
    # =========================================================================
    print("\n--- Test 3: Collision Rewards ---")
    env.reset()
    # P1 (Blocker) hits P2 (Enemy Runner)
    # Position them overlapping to trigger 1-tick collision
    teleport(1, torch.tensor([1000.0, 1000.0]))
    teleport(2, torch.tensor([1000.0, 1000.0])) # Perfect overlap
    
    # High Impact?
    env.physics.vel[0, 1] = torch.tensor([800.0, 0.0])
    env.physics.vel[0, 2] = torch.tensor([-800.0, 0.0]) # Head on
    
    ret = env.step(torch.zeros(4, 4, 4), weights)
    if len(ret) == 3: rew, _, _ = ret
    else: _, rew, _, _ = ret
    
    r_col_block = rew[0, 1].item()
    r_col_run = rew[0, 2].item()
    
    print(f"Blocker Collision Reward: {r_col_block:.2f} (Expected Positive)")
    print(f"Runner Collision Reward: {r_col_run:.2f} (Expected Small Positive/Neutral)")
    
    
    # =========================================================================
    # TEST 4: WIN REWARD (RUNNER)
    # =========================================================================
    print("\n--- Test 4: Win Reward (Runner) ---")
    env.reset()
    # Set P0 to last lap, last cp
    env.laps[0, 0] = 2 # 0-indexed, so 2 is Lap 3
    env.next_cp_id[0, 0] = 0 # Targetting Start (Finish Line)
    cp0 = env.checkpoints[0, 0]
    
    teleport(0, cp0)
    
    # Step
    ret = env.step(torch.zeros(4, 4, 4), weights)
    if len(ret) == 3: rew, _, _ = ret
    else: _, rew, _, _ = ret
    
    r_win = rew[0, 0].item()
    print(f"Runner Win Reward: {r_win:.2f} (Expected ~10000)")
    
    
    # =========================================================================
    # TEST 5: STEP PENALTY (BLOCKER MASK)
    # =========================================================================
    print("\n--- Test 5: Step Penalty (Blocker Mask) ---")
    env.reset()
    # Isolate Step Penalty by disabling Rank, Progress, etc.
    weights_iso = weights.clone()
    weights_iso[:, 13] = 0.0 # RW_RANK
    weights_iso[:, 4] = 0.0  # RW_PROGRESS
    weights_iso[:, 8] = 0.0  # RW_ORIENTATION
    weights_iso[:, 11] = 0.0 # PROXIMITY
    
    # Ensure P1 is Blocker
    env.is_runner[0, 1] = False
    
    # Move everyone apart (using teleport to sync)
    teleport(0, torch.tensor([0.0, 0.0]))
    teleport(1, torch.tensor([5000.0, 0.0]))
    teleport(2, torch.tensor([0.0, 5000.0]))
    teleport(3, torch.tensor([5000.0, 5000.0]))
    
    env.physics.vel[:] = 0.0
    
    # Step
    ret = env.step(torch.zeros(4, 4, 4), weights_iso)
    if len(ret) == 3: rew, _, _ = ret
    else: _, rew, _, _ = ret
    
    r_step_block = rew[0, 1].item()
    r_step_run = rew[0, 0].item()
    
    print(f"Blocker Step Penalty: {r_step_block:.2f} (Expected 0.0)")
    print(f"Runner Step Penalty: {r_step_run:.2f} (Expected -10.0)")

    # =========================================================================
    # TEST 6: RUNNER PROGRESS & ORIENTATION
    # =========================================================================
    print("\n--- Test 6: Runner Progress & Orientation ---")
    env.reset()
    
    # A. Progress (Movement)
    # Teleport P0 to origin, CP is at (Dist, 0)
    cp1_pos = env.checkpoints[0, 1] # Target
    teleport(0, torch.tensor([0.0, 0.0]))
    
    # Update Prev Dist manually to be "Start"
    env.prev_dist[0, 0] = torch.norm(cp1_pos - torch.tensor([0.0, 0.0]))
    
    # Move P0 towards CP1 at speed 100
    dir_to_cp = (cp1_pos - torch.tensor([0.0, 0.0]))
    dir_to_cp = dir_to_cp / torch.norm(dir_to_cp)
    env.physics.vel[0, 0] = dir_to_cp * 500.0
    
    # Orient P0 correctly
    angle = torch.atan2(dir_to_cp[1], dir_to_cp[0])
    env.physics.angle[0, 0] = torch.rad2deg(angle)
    
    # Weights: Enable Progress & Orient
    # Disable others to isolate
    w_prog = weights.clone()
    w_prog[:, 0] = 0 # Win
    w_prog[:, 2] = 0 # CP
    w_prog[:, 7] = 0 # Step (keep 0 to see pure gain, or keep it to see net?)
    # Let's keep Step Penalty to see Net Flow, or disable to verify component?
    # Disable Step Penalty to verify positive gradient clearly
    w_prog[:, 7] = 0 
    
    ret = env.step(torch.zeros(4, 4, 4), w_prog)
    if len(ret) == 3: rew, _, _ = ret
    else: _, rew, _, _ = ret
    
    r_run_prog = rew[0, 0].item()
    
    print(f"Runner Reward (Moving to CP, No Step Pen): {r_run_prog:.2f} (Expected Positive ~100)")
    
    # B. bad Orientation
    # Reset
    env.reset()
    teleport(0, torch.tensor([0.0, 0.0]))
    env.prev_dist[0, 0] = torch.norm(cp1_pos - torch.tensor([0.0, 0.0]))
    # Stationary, but facing wrong way
    env.physics.vel[0, 0] = torch.zeros(2)
    env.physics.angle[0, 0] = torch.rad2deg(angle) + 180.0
    
    ret = env.step(torch.zeros(4, 4, 4), w_prog)
    if len(ret) == 3: rew, _, _ = ret
    else: _, rew, _, _ = ret
    
    r_run_bad_orient = rew[0, 0].item()
    print(f"Runner Reward (Stationary, Bad Orient): {r_run_bad_orient:.2f} (Expected 0 or Low)")


if __name__ == "__main__":
    verify_all_rewards()
