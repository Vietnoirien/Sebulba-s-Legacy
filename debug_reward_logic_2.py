import torch
from simulation.env import PodRacerEnv
from config import EnvConfig, STAGE_TEAM, RW_WIN, RW_CHECKPOINT, RW_COLLISION_BLOCKER, RW_DENIAL 

# Helper to print non-zero rewards
def print_rewards(env, step_desc):
    print(f"\n--- {step_desc} ---")
    r_indiv = env.rewards.clone()
    print("Rewards Indiv (Last Step Accumulation):")
    headers = ["P0 (Run)", "P1 (Block)", "P2 (OppRun)", "P3 (OppBlock)"]
    vals = r_indiv[0].tolist()
    for h, v in zip(headers, vals):
        print(f"{h}: {v:.2f}")

def reproduction():
    print("Initializing Reward Debugger...")
    env = PodRacerEnv(num_envs=1, device='cpu')
    
    # Configure for Team Stage
    config = EnvConfig(mode_name="team", use_bots=False)
    env.set_stage(STAGE_TEAM, config, reset_env=True)
    
    # Force Roles
    # P0: Runner, P1: Blocker, P2: Runner, P3: Blocker
    env.is_runner[0, 0] = True
    env.is_runner[0, 1] = False
    env.is_runner[0, 2] = True
    env.is_runner[0, 3] = False
    
    print("Roles Set: P0=Run, P1=Block, P2=Run, P3=Block")
    
    # Reset Rewards
    env.rewards.fill_(0.0)
    
    actions = torch.zeros((1, 4, 4))
    
    # Scenario 1: Blocker (P1) hits CP
    print("\nScenario 1: Blocker (P1) ENTRIES Checkpoint 1")
    # Reset
    env.reset()
    env.is_runner[0, 0] = True; env.is_runner[0, 1] = False
    env.is_runner[0, 2] = True; env.is_runner[0, 3] = False
    env.rewards.fill_(0.0)
    
    # Create explicit reward weights
    # Size 20
    rw = torch.zeros((1, 20))
    rw[0, RW_CHECKPOINT] = 500.0
    rw[0, RW_WIN] = 10000.0
    rw[0, RW_DENIAL] = 10000.0
    rw[0, RW_COLLISION_BLOCKER] = 5.0
    rw[0, 16] = 5.0 # RW_ZONE (16)
    
    # Target CP1
    target = env.checkpoints[0, 1]
    
    # Radius is 600. Max Speed 800. 1 frame @ 60fps = 13.33 units.
    # Place at 605 units away. Move towards it.
    # End pos will be ~592 units. -> ENTRY.
    
    start_pos = target - torch.tensor([605.0, 0.0])
    env.physics.pos[0, 1] = start_pos
    env.physics.vel[0, 1] = torch.tensor([800.0, 0.0]) # Max speed East
    
    # [FIX] Move Opponents far away to avoid Proximity Reward noise
    env.physics.pos[0, 0] = torch.tensor([10000.0, 10000.0])
    env.physics.pos[0, 2] = torch.tensor([10000.0, 10000.0])
    env.physics.pos[0, 3] = torch.tensor([10000.0, 10000.0])
    
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    
    p1_rew = rewards[0, 1].item()
    if p1_rew > 100.0:
        print(f"FAIL: Blocker got CP Reward: {p1_rew}")
    else:
        print(f"PASS: Blocker CP Reward masked: {p1_rew}")

    # Scenario 2: Runner (P0) hits CP
    print("\nScenario 2: Runner (P0) ENTRIES Checkpoint 1")
    env.reset()
    env.is_runner[0, 0] = True; env.is_runner[0, 1] = False
    env.is_runner[0, 2] = True; env.is_runner[0, 3] = False
    env.rewards.fill_(0.0)
    
    target = env.checkpoints[0, 1]
    start_pos = target - torch.tensor([605.0, 0.0])
    env.physics.pos[0, 0] = start_pos
    env.physics.vel[0, 0] = torch.tensor([800.0, 0.0])
    
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    
    p0_rew = rewards[0, 0].item()
    if p0_rew > 100.0:
        print(f"PASS: Runner got CP Reward: {p0_rew}")
    else:
        print(f"FAIL: Runner got NO CP Reward: {p0_rew}")

    # Scenario 3: Opponent Runner (P2) hits CP
    print(f"\nScenario 3: Opponent Runner (P2) ENTRIES Checkpoint")
    env.reset()
    env.is_runner[0, 0] = True; env.is_runner[0, 1] = False # Team 0
    env.is_runner[0, 2] = True; env.is_runner[0, 3] = False # Team 1
    env.rewards.fill_(0.0)
    
    p2_next = env.next_cp_id[0, 2].item()
    target = env.checkpoints[0, p2_next]
    
    start_pos = target - torch.tensor([605.0, 0.0])
    env.physics.pos[0, 2] = start_pos
    env.physics.vel[0, 2] = torch.tensor([800.0, 0.0])
    
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    
    p0_rew = rewards[0, 0].item()
    p1_rew = rewards[0, 1].item()
    
    print(f"P0 (Runner) Reward: {p0_rew}")
    print(f"P1 (Blocker) Reward: {p1_rew}")
    
    # P1 (Blocker) should be penalized for Opponent P2 passing
    if p1_rew < -100.0:
        print("PASS: Blocker (P1) was penalized (Goalie Burden).")
    else:
        print("FAIL: Blocker (P1) was NOT penalized.")

    # Scenario 4: Wrong Way (Blocker)
    print("\nScenario 4: Blocker (P1) Wrong Way")
    env.reset()
    env.is_runner[0, 0] = True; env.is_runner[0, 1] = False
    env.rewards.fill_(0.0)
    
    # Position P1 facing AWAY from next CP
    next_cp = env.next_cp_id[0, 1].item()
    target = env.checkpoints[0, next_cp]
    
    env.physics.pos[0, 1] = torch.tensor([-2000.0, -2000.0]) # Far away
    # Checkpoint likely at (0,0) or similar? 
    # Let's explicitly set Target Pos-Current Pos
    # Vector To Target = Target - Pos
    # We want Angle diff > 90.
    # Set Pos=(0,0), Target=(1000,0). Angle 0.
    # Set P1 Angle = 180.
    env.checkpoints[0, next_cp] = torch.tensor([1000.0, 0.0])
    env.physics.pos[0, 1] = torch.tensor([0.0, 0.0])
    env.physics.angle[0, 1] = 180.0
    
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    p1_rew = rewards[0, 1].item()
    if p1_rew < -0.1:
         print(f"FAIL: Blocker (P1) penalized for Wrong Way: {p1_rew}")
    else:
         print(f"PASS: Blocker (P1) ignored Wrong Way: {p1_rew}")

    # Scenario 5: Timeout Denial
    print("\nScenario 5: Opponent Runner (P2) Timeout")
    env.reset()
    env.is_runner[0, 0] = True; env.is_runner[0, 1] = False
    env.is_runner[0, 2] = True; env.is_runner[0, 3] = False
    env.rewards.fill_(0.0)
    
    # Set P2 Timeout to 1
    env.timeouts[0, 2] = 1
    # Step to trigger timeout
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    
    p1_rew = rewards[0, 1].item()
    if p1_rew > 5000.0:
        print(f"PASS: Blocker (P1) got Denial Reward: {p1_rew}")
    else:
        print(f"FAIL: Blocker (P1) got NO Denial Reward: {p1_rew}")

    # Scenario 6: Win Assist
    print("\nScenario 6: Runner (P0) Wins Race")
    env.reset()
    env.is_runner[0, 0] = True; env.is_runner[0, 1] = False
    env.rewards.fill_(0.0)
    
    # Force Win
    env.laps[0, 0] = 3 # Max Laps
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    
    p0_rew = rewards[0, 0].item()
    p1_rew = rewards[0, 1].item()
    
    if p1_rew > 5000.0:
        print(f"PASS: Blocker (P1) got Win Assist: {p1_rew}")
    else:
        print(f"FAIL: Blocker (P1) got NO Win Assist: {p1_rew}")

def verify_stage2():
    print("\n\n=== VERIFYING STAGE 2 (DUEL) ===")
    from config import STAGE_DUEL_FUSED
    env = PodRacerEnv(num_envs=1, device='cpu')
    config = EnvConfig(mode_name="duel", use_bots=False)
    env.set_stage(STAGE_DUEL_FUSED, config, reset_env=True)
    
    # Explicit Reward Weights
    rw = torch.zeros((1, 20))
    rw[0, RW_CHECKPOINT] = 500.0
    rw[0, RW_WIN] = 10000.0
    rw[0, RW_DENIAL] = 10000.0
    
    actions = torch.zeros((1, 4, 4))
    
    # 1. Runner Win (Should work)
    print("\n[Stage 2] Runner (P0) Win")
    env.reset()
    env.is_runner[0, 0] = True; env.is_runner[0, 1] = False
    env.rewards.fill_(0.0)
    
    # Force Win
    env.laps[0, 0] = 3
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    
    p0_rew = rewards[0, 0].item()
    if p0_rew > 5000.0:
        print(f"PASS: Runner Win Reward: {p0_rew}")
    else:
        print(f"FAIL: Runner Win Reward Missing: {p0_rew}")

    # 2. Blocker Denial (Should work)
    print("\n[Stage 2] Blocker (P0) Denial")
    env.reset()
    env.is_runner[0, 0] = False # Blocker Agent
    env.is_runner[0, 1] = True  # Opponent (Bot) Runner [FIX]
    env.rewards.fill_(0.0)
    
    # Force Opponent Timeout (Bot)
    # In Duel, Bot is usually P2? Or P1?
    # Duel setup: P0 vs P1(Bot).
    # If P0 is Blocker, P0 denies P1.
    env.timeouts[0, 1] = 1 # Bot Timeout
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    
    p0_rew = rewards[0, 0].item()
    if p0_rew > 5000.0:
        print(f"PASS: Blocker Denial Reward: {p0_rew}")
    else:
        print(f"FAIL: Blocker Denial Reward Missing: {p0_rew}")

    # 3. Blocker Win (Should be 0)
    # If Agent Blocker (P0), and Agent Team Win?
    # Stage 2 is 1v1. If P0 is Blocker, and P0 'wins'? 
    # Logic: Blocker shouldn't get Race Win.
    print("\n[Stage 2] Blocker (P0) Race Win (Should Fail/Zero)")
    env.reset()
    env.is_runner[0, 0] = False
    env.rewards.fill_(0.0)
    env.laps[0, 0] = 3
    
    rewards, dones, infos = env.step(actions, reward_weights=rw)
    p0_rew = rewards[0, 0].item()
    
    if p0_rew < 100.0:
        print(f"PASS: Blocker got NO Win Reward: {p0_rew}")
    else:
        print(f"FAIL: Blocker got Win Reward (Regression): {p0_rew}")

    # 4. Collisions
    print("\n[Stage 2] Blocker Collision Metric")
    env.reset()
    env.is_runner[0, 0] = False
    
    # Setup Collision: P0 hits P2 (Enemy)
    env.physics.pos[0, 0] = torch.tensor([0.0, 0.0], device=env.device)
    # [FIX] Overlap to force immediate repulsion/impact
    env.physics.pos[0, 2] = torch.tensor([0.5, 0.0], device=env.device) 
    env.physics.vel[0, 0] = torch.tensor([10.0, 0.0], device=env.device) # Low speed
    env.physics.vel[0, 2] = torch.tensor([-10.0, 0.0], device=env.device) # Opposing
    
    # Reset Metrics
    env.stage_metrics["blocker_collisions"] = 0
    
    obs_unused, rew, done, infos = env.get_obs(), *env.step(actions, reward_weights=rw)
    # Wait, env.step returns (rew, done, info). 
    rew, done, infos = env.step(actions, reward_weights=rw)
    
    # Check Metric
    col_metric = env.stage_metrics.get("blocker_collisions", -1)
    if col_metric >= 1.0:
        print(f"PASS: Blocker Collision Metric Registered: {col_metric}")
    else:
        # Check impact to see if collision actually happened
        # impact = infos['blocker_damage'][0, 0].item() # Check legacy key too
        print(f"FAIL: Blocker Collision Metric Missing: {col_metric}")

if __name__ == "__main__":
    reproduction()
    verify_stage2()
