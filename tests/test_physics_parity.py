import unittest
import torch
import random
import math
from simulation.cpu_physics import Pod, Checkpoint, step as cpu_step, Point
from simulation.gpu_physics import GPUPhysics

class TestPhysicsParity(unittest.TestCase):
    def test_parity(self):
        # 1. Setup Initial State
        num_envs = 1
        device = 'cpu' # Use CPU for torch to make comparison easier/exact floats
        gpu_sim = GPUPhysics(num_envs, device=device)
        
        # Randomize positions
        pods_cpu = []
        for i in range(4):
            x = random.uniform(2000, 14000)
            y = random.uniform(2000, 7000)
            angle = random.uniform(-180, 180)
            vx = random.uniform(-100, 100)
            vy = random.uniform(-100, 100)
            
            # CPU Pod
            p = Pod(i, i//2, x, y, angle)
            p.vx = vx
            p.vy = vy
            pods_cpu.append(p)
            
            # GPU Pod
            gpu_sim.pos[0, i, 0] = x
            gpu_sim.pos[0, i, 1] = y
            gpu_sim.vel[0, i, 0] = vx
            gpu_sim.vel[0, i, 1] = vy
            gpu_sim.angle[0, i] = angle

        # 2. Run Steps
        num_steps = 10
        for step in range(num_steps):
            # Generate random actions
            actions_thrust = torch.zeros((1, 4))
            actions_angle = torch.zeros((1, 4))
            actions_shield = torch.zeros((1, 4), dtype=torch.bool)
            actions_boost = torch.zeros((1, 4), dtype=torch.bool)

            for i in range(4):
                thrust = random.randint(0, 100)
                angle_off = random.uniform(-18.0, 18.0)
                
                # CPU Apply
                pods_cpu[i].next_thrust = thrust
                pods_cpu[i].next_angle = pods_cpu[i].angle + angle_off # Simplified logic
                
                # GPU Apply
                actions_thrust[0, i] = thrust
                actions_angle[0, i] = angle_off
            
            # CPU Step
            # Need checkpoints dummy
            checkpoints = [Checkpoint(0, 0, 0)]
            cpu_step(pods_cpu, checkpoints)
            
            # GPU Step
            gpu_sim.step(actions_thrust, actions_angle, actions_shield, actions_boost)
            
            # 3. Compare
            for i in range(4):
                p = pods_cpu[i]
                
                # Pos
                gx = gpu_sim.pos[0, i, 0].item()
                gy = gpu_sim.pos[0, i, 1].item()
                
                # Vel
                gvx = gpu_sim.vel[0, i, 0].item()
                gvy = gpu_sim.vel[0, i, 1].item()
                
                # Check closeness
                # Note: Floating point differences may accumulate.
                # CPU uses python float (double precision usually), PyTorch defaults float32 unless specified.
                # If discrepancies occur, check precision.
                
                self.assertAlmostEqual(p.x, gx, delta=1.0, msg=f"Step {step} Pod {i} X mismatch")
                self.assertAlmostEqual(p.y, gy, delta=1.0, msg=f"Step {step} Pod {i} Y mismatch")
                self.assertAlmostEqual(p.vx, gvx, delta=1.0, msg=f"Step {step} Pod {i} VX mismatch")
                self.assertAlmostEqual(p.vy, gvy, delta=1.0, msg=f"Step {step} Pod {i} VY mismatch")

if __name__ == '__main__':
    unittest.main()
