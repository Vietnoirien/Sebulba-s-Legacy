
import torch
from simulation.env import PodRacerEnv

def test_start_positions():
    print("Initializing environment...")
    env = PodRacerEnv(num_envs=5, device='cpu') # Use CPU for simple print debugging
    
    print("Resetting environment...")
    env.reset()
    
    print("Checking start positions (Pod 0):")
    for i in range(5):
        pos = env.physics.pos[i, 0]
        cp0 = env.checkpoints[i, 0]
        print(f"Env {i}: Pos: {pos.tolist()}, CP0: {cp0.tolist()}")
        
    print("\nResetting again to check randomness...")
    env.reset()
    for i in range(5):
        pos = env.physics.pos[i, 0]
        print(f"Env {i} (Run 2): Pos: {pos.tolist()}")

if __name__ == "__main__":
    test_start_positions()
