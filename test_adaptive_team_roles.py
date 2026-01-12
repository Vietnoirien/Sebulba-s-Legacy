import torch
from simulation.env import PodRacerEnv, STAGE_TEAM

def test_adaptive_roles():
    print("Initializing Env in STAGE_TEAM...")
    env = PodRacerEnv(num_envs=2, device='cuda', start_stage=STAGE_TEAM)
    env.reset()
    
    print("\n--- Initial State ---")
    # At reset, dists are equal. Tie-breaker might apply.
    # Logic: s0 >= s1 -> Pod0 Runner. s2 >= s3 -> Pod2 Runner.
    # Pod 0 and 2 should be Runners initially.
    r = env.is_runner
    print(f"Roles: {r.int()}")
    
    # Assert Defaults
    # Team 0
    assert r[0, 0] == 1, "Pod 0 should be Runner (Tie-break)"
    assert r[0, 1] == 0, "Pod 1 should be Blocker"
    # Team 1
    assert r[0, 2] == 1, "Pod 2 should be Runner (Tie-break)"
    assert r[0, 3] == 0, "Pod 3 should be Blocker"
    
    print("PASS: Initial Roles Correct.")
    
    print("\n--- Scenario: Pod 1 Overtakes Pod 0 ---")
    # Manipulate Progress
    # Set Pod 1 ahead (e.g. Next CP + 1)
    env.next_cp_id[:, 1] = 2
    env.next_cp_id[:, 0] = 1
    
    # Force update (usually happens in reset or step)
    # We can call internal method
    indices = torch.arange(2, device='cuda')
    env.role_lock_timer[:] = 0 # Ensure no lock
    
    # We need to set prev_dist to calculate score correctly
    env.prev_dist[:] = 1000.0 
    
    env._update_roles(indices)
    
    r_new = env.is_runner
    print(f"New Roles: {r_new.int()}")
    
    # Expect Pod 1 to be Runner
    assert r_new[0, 1] == 1, "Pod 1 should swap to Runner (Leading)"
    assert r_new[0, 0] == 0, "Pod 0 should swap to Blocker"
    print("PASS: Swap Team 0 Successful.")

    print("\n--- Scenario: Hysteresis Lock ---")
    # Verify Lock is set
    timers = env.role_lock_timer
    print(f"Lock Timers: {timers}")
    assert (timers > 0).all(), "Role Change should trigger Hysteresis Lock"
    
    # Try to swap back immediately (Pod 0 huge jump)
    env.next_cp_id[:, 0] = 5
    # Should NOT update because lock > 0
    env._update_roles(indices)
    
    r_locked = env.is_runner
    print(f"Locked Roles: {r_locked.int()}")
    
    # Should stay as [0, 1]
    assert r_locked[0, 1] == 1, "Pod 1 should remain Runner due to Lock"
    print("PASS: Hysteresis Lock Successful.")
    
    print("\n--- Rank Observation Verification ---")
    # User asked: "model should determine who is the opponent leader by his rank"
    # Verify ranks are in entity observation (index -2 usually, check get_obs)
    
    self_obs, tm_obs, en_obs, cp_obs, map_obs = env.get_obs()
    
    # rank is index 11 in Entity Obs (last is timeout, before that rank)
    # Check self obs rank (Index 11 in self keys?)
    # Self: v_local(2), t_vec_l(2), dest(1), align(2), shield(1), boost(1), timeout(1), lap(1), leader(1), v_mag(1), pad(1), rank(1)
    # Total 15 dims.
    # Rank is last (-1).
    
    # Let's check ranges
    # self_obs is [4, B, 14]
    # Rank is the last feature (-1).
    # env.get_obs returns [4, B, 14] for self_c
    
    # Pod 1 is leading (Rank 0/3 -> 0.0)
    # Pod 0 is behind (Rank 1/3 -> 0.33)
    
    # Access via permuted tensor
    # self_c shape is [4, B, 14]
    # Pod 1 is index 1.
    p1_rank = self_obs[1, 0, -1].item()
    p0_rank = self_obs[0, 0, -1].item()
    
    print(f"Pod 1 Rank (Leader): {p1_rank:.2f}")
    print(f"Pod 0 Rank (Chaser): {p0_rank:.2f}")
    
    assert p1_rank < p0_rank, "Leader should have lower rank value (0=First)"
    
    print("PASS: Rank Observation Correct.")

if __name__ == "__main__":
    try:
        test_adaptive_roles()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
