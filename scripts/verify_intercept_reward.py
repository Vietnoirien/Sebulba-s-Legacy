
import torch
import numpy as np
import sys
import os

# Adjust path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.env import PodRacerEnv, STAGE_DUEL_FUSED
from config import TrainingConfig, EnvConfig, BotConfig, RewardScalingConfig

def verify_intercept_reward():
    print("=== Verifying Blocker Intercept Progress Reward ===")
    
    # 1. Setup Env (Duel Stage to activate Blocker Logic)
    # We use a custom config to isolate Intercept Reward if possible, 
    # but the metric 'blocker_intercept_metric' already isolates it! Perfect.
    
    t_config = TrainingConfig(num_envs=1, device='cpu') # CPU for debugging
    curriculum_config = None # Default
    
    # Enable Intercept Reward Scale
    rew_config = RewardScalingConfig(
        intercept_progress_scale=1.0 # Set to 1.0 to see raw values (or keeping default 0.05 is fine, we just check sign)
    )
    
    env_config = EnvConfig(
        mode_name="duel",
        use_bots=True,
        bot_pods=[2],
        fixed_roles={0: 0, 1: 0, 2: 1, 3: 0}, # Pod 0 is Blocker (0), Pod 2 is Runner (1)
        active_pods=[0, 2]
    )
    
    # Correct Instantiation
    # env = PodRacerEnv(num_envs, device, start_stage)
    env = PodRacerEnv(num_envs=t_config.num_envs, device=t_config.device, start_stage=STAGE_DUEL_FUSED)
    
    # Manually Inject Configs
    env.config = env_config
    env.reward_scaling_config = rew_config
    
    # Re-init Physics if needed (usually handled by reset, but num_envs is same)
    # Env init creates physics.
    
    # Reset to apply config (e.g. active pods)
    obs = env.reset()
    
    # 2. Setup Scenario
    # Enemy (Pod 2): Moving Right towards CP
    # Next CP for Pod 2 needs to be known.
    # We can force positions.
    
    # Force Env State
    env.physics.pos[:] = 0.0
    env.physics.vel[:] = 0.0
    env.physics.angle[:] = 0.0
    
    # Enemy (Pod 2)
    # Pos: (10000, 0)
    # Vel: (500, 0) -> Moving Right
    # Next CP: Let's assume CP is at (15000, 0).
    # We need to find which CP is "Next".
    # We will override checkpoints to be simple.
    
    # Set Checkpoints: CP 0 at (0,0), CP 1 at (15000, 0)
    env.checkpoints[0, 0] = torch.tensor([0.0, 0.0])
    env.checkpoints[0, 1] = torch.tensor([15000.0, 0.0])
    env.next_cp_id[0, 2] = 1 # Enemy aiming for CP 1
    
    env.physics.pos[0, 2] = torch.tensor([10000.0, 0.0])
    env.physics.vel[0, 2] = torch.tensor([500.0, 0.0])
    
    # Calculated Intercept Point
    # Agent Pos: (0, 0)
    # Agent Vel: (0, 0)
    # Env calculates intercept.
    # Logic: "Gatekeeper": 2500u in front of CP (Low Diff) or 500u (High Diff).
    # Default Diff is 0.0 -> Offset 2500.
    # CP is (15000, 0). Intercept should be approx (12500, 0).
    
    print(f"Enemy Pos: {env.physics.pos[0, 2].tolist()}")
    print(f"Enemy Vel: {env.physics.vel[0, 2].tolist()}")
    print(f"Enemy Target CP: {env.checkpoints[0, 1].tolist()}")
    print(f"Expected Intercept (Diff 0): ~12500.0, 0.0 (2500u before CP)")
    
    # Call internal helper to verify calculation
    # _get_front_intercept_pos(curr_pos, enemy_pos, enemy_vel, me_vel)
    # It uses 'self.bot_difficulty' which defaults to 0.0
    
    calc_intercept = env._get_front_intercept_pos(
        env.physics.pos[0, 0].unsqueeze(0),
        env.physics.pos[0, 2].unsqueeze(0),
        env.physics.vel[0, 2].unsqueeze(0),
        env.physics.vel[0, 0].unsqueeze(0)
    ).squeeze(0)
    
    print(f"Calculated Intercept Point: {calc_intercept.tolist()}")
    
    # 3. Test Case A: Move Towards Intercept
    # Agent at (5000, 0). Intercept at (12500, 0).
    # Vector To Intercept: (+7500, 0)
    # Velocity: (+600, 0) -> Positive Reward
    
    env.physics.pos[0, 0] = torch.tensor([5000.0, 0.0])
    env.physics.vel[0, 0] = torch.tensor([600.0, 0.0]) # Moving Right (Correct)
    
    # Step (Actions don't matter much as we override physics, but we need step to trigger reward calc)
    # NOTE: Step UPDATES physics. So our override might be overwritten if we don't be careful.
    # Actually step() applies physics then calcs reward.
    # So we should apply ACTION that creates velocity? 
    # Or just override physics AFTER step? No, reward is calc'd on CURRENT state in step().
    # Wait, look at env.py:
    # 1. Physics Step updates pos/vel.
    # 2. Rewards calculated based on NEW pos/vel.
    
    # So we should set state, then call a dummy "calculate_rewards" or just mock the loop.
    # Creating a mock loop is hard.
    # Easier: Set state such that AFTER physics step it is what we want.
    # If we set velocity, inertia keeps it mostly same for one step (friction 0.85).
    
    actions = torch.zeros((1, 4, 4))
    # obs, rewards, dones, infos = env.step(actions) -> rewards, dones, infos = env.step(actions, None)
    rewards, dones, infos = env.step(actions, None)
    
    intercept_rew_A = infos['blocker_intercept'][0, 0].item()
    print(f"\nTest A (Moving TOWARDS Intercept):")
    print(f"Agent Pos: {env.physics.pos[0, 0].tolist()}")
    print(f"Agent Vel: {env.physics.vel[0, 0].tolist()}")
    print(f"Reward (Blocker Intercept Metric): {intercept_rew_A:.4f}")
    
    if intercept_rew_A > 0.0:
        print("✅ SUCCESS: Positive reward for moving towards intercept.")
    else:
        print("❌ FAILURE: Expected positive reward.")
        
    # 4. Test Case B: Move AWAY from Intercept
    # Agent at (5000, 0). Intercept at (12500, 0).
    # Velocity: (-600, 0) -> Moving Left (Wrong)
    
    # Reset / Force State again
    env.physics.pos[0, 0] = torch.tensor([5000.0, 0.0])
    env.physics.vel[0, 0] = torch.tensor([-600.0, 0.0]) # Moving Left (Away)
    
    # obs, rewards, dones, infos = env.step(actions, None)
    rewards, dones, infos = env.step(actions, None)
    
    intercept_rew_B = infos['blocker_intercept'][0, 0].item()
    print(f"\nTest B (Moving AWAY FROM Intercept):")
    print(f"Agent Pos: {env.physics.pos[0, 0].tolist()}")
    print(f"Agent Vel: {env.physics.vel[0, 0].tolist()}")
    print(f"Reward (Blocker Intercept Metric): {intercept_rew_B:.4f}")
    
    if intercept_rew_B < 0.0:
        print("✅ SUCCESS: Negative reward for moving away from intercept.")
    else:
        print("❌ FAILURE: Expected negative (or lower) reward.")

    # 5. Check Magnitude
    # Velocity Project logic: v_proj * scale * dense_mult
    # v_proj ~ 600 * 0.85 (friction) ~ 510.
    # Scale = 1.0. Dense Mult ~ 0.5 (Tau starts at 0.5 in env?) 
    # Env Tau isn't passed in Step. It's a global var in step() in code?
    # No, tau is passed to dense_mult calculation using self.curriculum or config?
    # In env.py: `dense_mult = (1.0 - tau)`. `tau` is usually a parameter or property.
    # In Env.step(), `tau` is not an argument. It's usually `self.tau`?
    # Checking env.py... `dense_mult = (1.0 - tau)`. Where does tau come from?
    # Ah, lines 806: `dense_mult = (1.0 - tau)`. `tau` is NOT defined locally!
    # It must be a global config or self member?
    # Error in env.py logic? 
    # Checking env.py code snippet... 
    # PyTorch Env usually has `tau` passed in `update_config` or similar. I don't see `self.tau`.
    # Maybe it relies on `tau` being in global scope (Bad) or `config.tau`.
    # Wait, `training/session.py` sets `self.env.tau = ...`?
    # Let's check if verify script crashes on 'tau'.
    
    if intercept_rew_A > 0 and intercept_rew_B < 0:
        print("\n🏆 VERIFICATION COMPLETE: Intercept Progress Reward Works.")
    else:
        print("\n⚠️ VERIFICATION FAILED.")

if __name__ == "__main__":
    verify_intercept_reward()
