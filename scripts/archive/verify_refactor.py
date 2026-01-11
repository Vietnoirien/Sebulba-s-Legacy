
import torch
from simulation.env import PodRacerEnv, STAGE_DUEL
from config import TrainingConfig, CurriculumConfig

def verify():
    print("--- Verifying Refactor ---")
    try:
        # 1. Config Check
        print("Checking Config Structures...")
        t_conf = TrainingConfig()
        c_conf = CurriculumConfig()
        print(f"Training Config Loaded. Penalty Const: {t_conf.proficiency_penalty_const}")
        print(f"Curriculum Config Loaded. WR Critical: {c_conf.wr_critical}")

        # 2. Env Init
        print("Initializing Env...")
        env = PodRacerEnv(16, device='cpu', start_stage=STAGE_DUEL)
        print(f"Env Loaded. Bot Config Noise: {env.bot_config.difficulty_noise_scale}")
        
        # 3. Step Check
        print("Stepping Env...")
        actions = torch.zeros((16, 4, 4))
        env.reset()
        env.step(actions, None)
        print("Step Successful.")
        
        print("--- VERIFICATION PASSED ---")
        
    except Exception as e:
        print(f"--- VERIFICATION FAILED ---")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
