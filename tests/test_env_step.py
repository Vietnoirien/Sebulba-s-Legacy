
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
    self_obs, tm_obs, en_obs, cp_obs = obs
    print(f"Self: {self_obs.shape}, Tm: {tm_obs.shape}, En: {en_obs.shape}, CP: {cp_obs.shape}")


    print("Stepping...")
    actions = torch.zeros((4, 4, 4)) # [B, 4, 4]
    
    rewards, dones, infos = env.step(actions, None)
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
