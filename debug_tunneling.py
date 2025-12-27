
import torch
import sys
import os

sys.path.append(os.getcwd())
from simulation.env import PodRacerEnv

def test_tunneling():
    device = 'cpu'
    env = PodRacerEnv(1, device=device)
    env.reset()
    
    # Check Init
    print(f"Init Next CP: {env.next_cp_id[0]}")
    print(f"Num CPs: {env.num_checkpoints[0]}")
    print(f"Checkpoints: {env.checkpoints[0]}")
    
    # 1. Setup
    cp1 = env.checkpoints[0, 1]
    
    # Move Pod 0 to start
    env.physics.pos[0, 0, 0] = cp1[0] - 450
    env.physics.pos[0, 0, 1] = cp1[1] + 500
    env.physics.vel[0, 0, 0] = 900 
    
    print(f"Start Pos: {env.physics.pos[0, 0]}")
    
    # 3. Step
    actions = torch.zeros((1, 4, 4))
    rewards, dones, _ = env.step(actions, None)
    
    # 4. Results
    print(f"Final Pos: {env.physics.pos[0, 0]}")
    print(f"Next CP After Step: {env.next_cp_id[0]}")
    
    # Manual check
    if env.next_cp_id[0, 0] == 2:
        print("SUCCESS: Tunneling Detected!")
    elif env.next_cp_id[0, 0] == 1:
        print("FAILURE: Missed Checkpoint (Still 1).")
    else:
        print(f"FAILURE: Unexpected CP {env.next_cp_id[0, 0]}")

if __name__ == "__main__":
    test_tunneling()
