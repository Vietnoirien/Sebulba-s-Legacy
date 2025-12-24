import torch
from simulation.env import PodRacerEnv
from config import *

def test_timeout_reset():
    print("Testing environment timeout reset...")
    num_envs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Start in Solo stage
    env = PodRacerEnv(num_envs, device=device, start_stage=STAGE_SOLO)
    
    # Take TIMEOUT_STEPS steps without finishing
    for i in range(TIMEOUT_STEPS):
        # Zero actions (stay still)
        actions = torch.zeros((num_envs, 4, 4), device=device)
        rewards, dones = env.step(actions)
        
        if i < TIMEOUT_STEPS - 1:
            assert not dones.any(), f"Env done prematurely at step {i}"
        else:
            assert dones.all(), f"Env didn't timeout at step {i}"
            
    print("Timeout reset verified successfully!")

if __name__ == "__main__":
    test_timeout_reset()
