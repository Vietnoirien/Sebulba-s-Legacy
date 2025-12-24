
import torch
from simulation.env import PodRacerEnv, STAGE_SOLO

def test_env():
    print("Initializing Env...")
    env = PodRacerEnv(num_envs=4, device='cpu', start_stage=STAGE_SOLO)
    print("Resetting...")
    env.reset()
    
    print("Getting Obs...")
    obs = env.get_obs()
    # obs is list of 4 tuples (self, entity, cp)
    print(f"Obs Len: {len(obs)}")
    for i, (s, e, c) in enumerate(obs):
        print(f"Pod {i}: Self {s.shape}, Ent {e.shape}, CP {c.shape}")
        assert s.shape == (4, 14)
        assert e.shape == (4, 3, 13)
        assert c.shape == (4, 6)
        
        # Check for NaNs
        if torch.isnan(s).any(): print(f"NaN in Self Obs {i}")
        if torch.isnan(e).any(): print(f"NaN in Ent Obs {i}")
        if torch.isnan(c).any(): print(f"NaN in CP Obs {i}")

    print("Stepping...")
    actions = torch.zeros((4, 4, 4)) # [B, 4, 4]
    
    rewards, dones, infos = env.step(actions)
    print("Step 1 done. Rewards shape:", rewards.shape)
    print("Rewards:", rewards)
    
    if torch.isnan(rewards).any():
        print("NaN rewards detected!")
    else:
        print("Rewards are valid numbers.")

if __name__ == "__main__":
    try:
        test_env()
        print("Test Success")
    except Exception as e:
        print("Test Failed (Exception):")
        print(e)
        import traceback
        traceback.print_exc()
