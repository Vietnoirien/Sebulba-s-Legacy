import torch
import sys
import os
import time

# Add root to path
sys.path.append(os.getcwd())

from training.ppo import PPOTrainer
from config import STAGE_INTERCEPT

def verify_metrics():
    print("Initializing Trainer...")
    trainer = PPOTrainer(device='cuda')
    
    # 1. Force Stage 3
    print("Forcing Stage 3 (Intercept)...")
    trainer.curriculum.set_stage(STAGE_INTERCEPT)
    trainer.env.set_stage(STAGE_INTERCEPT, trainer.curriculum.current_stage.get_env_config(), reset_env=True)
    
    # 2. Force Difficulty to 1.0 (Verification of Perfect Trajectory Setting)
    print(f"Checking Bot Difficulty: {trainer.env.bot_difficulty}")
    # Note: stages.py update() ensures this, but we need to call it or manually set it if we skipped loop.
    trainer.curriculum.current_stage.update(trainer)
    print(f"Bot Difficulty after update: {trainer.env.bot_difficulty}")
    assert trainer.env.bot_difficulty == 1.0, "Bot Difficulty should be 1.0 in Stage 3"
    
    # 3. Verify Denial Metric Logic
    print("Simulating Denial Condition (Runner Timeout)...")
    trainer.env.reset()
    
    # Force Runner (Pod 2) to be near timeout
    # trainer.env.timeouts is [Envs, 4]
    trainer.env.timeouts[:, 2] = 2 # 2 steps remaining
    
    # Step 1
    actions = torch.zeros((trainer.config.num_envs, 4, 4), device=trainer.device)
    # Give some thrust to ensure no idle penalty
    actions[:, :, 2] = 100.0 
    
    print("Stepping Env (1/2)...")
    # Provide required args: reward_weights, tau=0, team_spirit=0
    trainer.env.step(actions, reward_weights=trainer.reward_weights_tensor, tau=0.0, team_spirit=0.0)
    
    print("Stepping Env (2/2) - Expect Timeout...")
    rewards, dones, infos = trainer.env.step(actions, reward_weights=trainer.reward_weights_tensor, tau=0.0, team_spirit=0.0)
    
    # Check if Dones triggered
    num_dones = dones.sum().item()
    print(f"Dones Triggered: {num_dones} / {trainer.config.num_envs}")
    assert num_dones == trainer.config.num_envs, "All envs should timeout"
    
    # Check Winners
    winners = trainer.env.winners # [Envs]
    print(f"Winners sum (+1 for Team 1, 0 for Team 0, -1 for None): {winners.float().sum().item()}")
    # Should be all -1 (or whatever 'No Winner' is. DType is int usually, initialized to -1)
    
    # Verify Win Reward Masking
    # Check rewards for Blocker (Pod 0). Should be negative (Time Penalty only) or near zero, definitely NO Win Reward.
    # Note: Win Reward is applied to rewards_team.
    # If unmasked, Team 0 would get Win Reward if they won. But here nobody won.
    # Let's force a "Win" condition artificially later to check masking.
    
    # [NEW] Check Denial Reward
    # We expect Blocker (Pod 0) to have received a large positive reward (w_win or 5000).
    # Since we used w_win default (likely 10,000) or 5000.
    r_blocker_denial = rewards[:, 0].mean().item()
    print(f"Blocker Reward (Denial Condition): {r_blocker_denial}")
    assert r_blocker_denial > 4000.0, f"Blocker should receive Denial Reward (>4000.0), got {r_blocker_denial}"

    # Verify Denial Count in Population (Mocking the PPO Loop Logic)
    # Logic in PPO:
    # denials = (done_mask & (w_reshaped == -1)).float().sum(dim=1)
    
    w_reshaped = winners.view(trainer.config.pop_size, trainer.config.envs_per_agent)
    d_reshaped = dones.view(trainer.config.pop_size, trainer.config.envs_per_agent)
    
    denials = (d_reshaped & (w_reshaped == -1)).float().sum().item()
    print(f"Total Denials Calculated: {denials}")
    assert denials == trainer.config.num_envs, "Should equal total envs (all timed out)"
    
    # 4. Verify Win Reward Masking
    print("Simulating Artificial Blocker Win to check Reward Masking...")
    trainer.env.reset()
    trainer.env.laps[:, 0] = 5 # Force Pod 0 (Blocker) to finish
    
    # Step to trigger finish logic
    rewards, dones, infos = trainer.env.step(actions, reward_weights=trainer.reward_weights_tensor, tau=0.0, team_spirit=0.0)
    
    # Blocker (Pod 0) Reward check
    # Reward should NOT include WIN_REWARD (usually 10,000+)
    # It might include Lap Reward (2000 * 3^...)
    # We masked RW_WIN in env.py lines 1121.
    # We masked RW_LAP? No, lines 967 masking 'total_reward' for blockers if INTERCEPT.
    # Wait, line 967: `total_reward = total_reward * is_run_mask`.
    # This masks LAP and CHECKPOINT rewards for Blocker.
    # So Blocker should get ZERO progress/lap rewards.
    
    r_blocker = rewards[:, 0].mean().item()
    print(f"Blocker Reward (Win Condition): {r_blocker}")
    # Should be close to zero (maybe collision penalties exists, but here no collisions).
    # Definitely not 10,000+.
    assert r_blocker < 100.0, f"Blocker Reward ({r_blocker}) too high! Win/Lap rewards not masked."
    
    print(">>> VERIFICATION SUCCESSFUL <<<")

if __name__ == "__main__":
    verify_metrics()
