
import torch
from simulation.env import PodRacerEnv, STAGE_DUEL
from training.curriculum.stages import DuelStage
from config import CurriculumConfig

def debug_roles():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = PodRacerEnv(num_envs=4, device=device)
    
    # Setup Duel Stage Config
    curr_config = CurriculumConfig()
    stage = DuelStage(curr_config)
    env_config = stage.get_env_config()
    
    # Set Stage
    env.set_stage(STAGE_DUEL, env_config, reset_env=True)
    
    # Check Pod Positions
    pos = env.physics.pos
    print(f"Pod 0 Pos: {pos[0, 0]}")
    print(f"Pod 1 Pos: {pos[0, 1]}") # Should be infinity
    
    # Check Active Pods
    print(f"Config Active Pods: {env.config.active_pods}")
    
    # Check Prev Dist to CP1
    print(f"Prev Dist 0: {env.prev_dist[0, 0]}")
    print(f"Prev Dist 1: {env.prev_dist[0, 1]}")
    
    # Check Roles
    print(f"Is Runner 0: {env.is_runner[0, 0]}")
    print(f"Is Runner 1: {env.is_runner[0, 1]}")
    
    # Force Step to trigger update_roles check
    env.role_lock_timer[:] = 0
    env._update_roles(torch.arange(4, device=device))
    
    print("--- After Update Roles ---")
    print(f"Is Runner 0: {env.is_runner[0, 0]}")
    print(f"Is Runner 1: {env.is_runner[0, 1]}")

if __name__ == "__main__":
    debug_roles()
