import torch
import torch.nn as nn
from simulation.env import PodRacerEnv, STAGE_DUEL_FUSED

# Mock Config
class MockConfig:
    def __init__(self):
        self.num_envs = 64
        self.use_cuda = True
        from config import EnvConfig, STAGE_DUEL_FUSED
        
        self.curriculum = {
            STAGE_DUEL_FUSED: EnvConfig(
                 mode_name="duel",
                 track_gen_type="max_entropy",
                 active_pods=[0, 2],
                 use_bots=True,
                 bot_pods=[2],
                 step_penalty_active_pods=[0],
                 orientation_active_pods=[0]
            )
        }

def test_stage2_metric_update():
    print("=== Testing Stage 2 Metric Update Logic ===")
    
    cfg = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Env
    env = PodRacerEnv(cfg.num_envs, device=device, start_stage=STAGE_DUEL_FUSED)
    stage_config = cfg.curriculum[STAGE_DUEL_FUSED]
    env.set_stage(STAGE_DUEL_FUSED, stage_config, reset_env=True)
    
    print("Verifying Role Assignment...")
    is_run_group_a = env.is_runner[0, 0]
    is_run_group_b = env.is_runner[32, 0]
    print(f"Group A (Env 0) Agent 0 Runner? {is_run_group_a}")
    print(f"Group B (Env 32) Agent 0 Runner? {is_run_group_b}")
    assert is_run_group_a == True
    assert is_run_group_b == False
    
    # Force Collision
    print("Forcing Collision in Group B...")
    env.physics.pos[32:, 0] = torch.tensor([5000.0, 5000.0], device=device)
    env.physics.pos[32:, 2] = torch.tensor([5000.0, 5000.0], device=device)
    
    env.physics.vel[32:, 0] = torch.tensor([100.0, 0.0], device=device)
    env.physics.vel[32:, 2] = torch.tensor([-100.0, 0.0], device=device)
    
    # Actions must be [B, 4, 4] (Batch, Pods, Channels) to match act_angle logic
    actions = torch.zeros((64, 4, 4), device=device)
    
    print("Stepping environment...")
    from config import DEFAULT_REWARD_WEIGHTS
    rw = torch.tensor(list(DEFAULT_REWARD_WEIGHTS.values()), device=device).unsqueeze(0).expand(64, -1)
    # Actually env.step expects tensor [Batch, Weights]
    # Wait, values are dict, we need ordered tensor based on indices
    # config.py defines RW_WIN=0, etc.
    # We should construct it properly.
    
    vals = [0.0] * 20
    for k, v in DEFAULT_REWARD_WEIGHTS.items():
        if k < 20: vals[k] = v
    rw_tensor = torch.tensor(vals, device=device).unsqueeze(0).expand(64, -1)
    
    obs, rew, dones, infos = env.get_obs(), *env.step(actions, reward_weights=rw_tensor)
    # env.step returns (rewards, dones, infos). 
    # obs must be called explicitly if needed, typically before or after?
    # Actually PPO calls step, then reset_dones, then get_obs.
    # We just need step to update metrics.
    # So:
    rew, dones, infos = env.step(actions, reward_weights=rw_tensor)
    
    collisions = env.stage_metrics.get("blocker_collisions", 0)
    print(f"Blocker Collisions Metric: {collisions}")
    
    assert collisions > 0, "Metric should be > 0 after forced collision!"
    assert collisions == 32, f"Expected 32 collisions, got {collisions}"
    
    print("✅ Metric Logic Verified: Collisions are correctly tracked.")

if __name__ == "__main__":
    test_stage2_metric_update()
