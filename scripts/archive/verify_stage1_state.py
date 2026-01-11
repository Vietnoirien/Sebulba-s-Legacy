import torch
from simulation.env import PodRacerEnv
from config import TrainingConfig, STAGE_SOLO

def verify():
    print("--- Verifying Stage 1 State ---")
    cfg = TrainingConfig()
    cfg.num_envs = 4
    env = PodRacerEnv(4, device='cpu')
    env.curriculum_stage = STAGE_SOLO
    env.reset()
    
    print(f"Stage: {env.curriculum_stage} (Expected {STAGE_SOLO})")
    print(f"Active Pods: {env.config.active_pods} (Expected [0])")
    print(f"Is Runner Tensor:\n{env.is_runner}")
    
    # Check consistency
    is_r_0 = env.is_runner[:, 0].float().mean().item()
    print(f"Pod 0 Is Runner Mean: {is_r_0} (Expected 1.0)")
    
    # Simulate PPO Logic
    # If done=True, Winner=0.
    done_mask = torch.tensor([True]*4)
    winners = torch.zeros(4, dtype=torch.long)
    
    wins = (done_mask & (winners == 0)).float().sum()
    print(f"Global Wins: {wins}")
    
    active_pods = env.config.active_pods
    runner_matches = 0
    
    for pod_idx in active_pods:
        is_r = env.is_runner[:, pod_idx].float()
        rm = (done_mask.float() * is_r).sum()
        runner_matches += rm
        
    print(f"Runner Matches: {runner_matches}")
    
    if runner_matches == 0:
        print("FAIL: Runner Matches is 0. Division by Zero or Inf result.")
    elif wins > runner_matches:
        print(f"FAIL: Wins ({wins}) > Matches ({runner_matches}). Rate {wins/runner_matches}")
    else:
        print(f"PASS: Rate {wins/runner_matches}")

if __name__ == "__main__":
    verify()
