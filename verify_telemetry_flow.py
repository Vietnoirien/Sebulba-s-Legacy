
import torch
import struct
import base64
import numpy as np
from simulation.env import PodRacerEnv

def pack_frame_mock(pos, vel, angle, laps, next_cps, rewards, actions, step, env_idx):
    # Mocking telemetry.pack_frame logic
    header = struct.pack('<HBIH7x', 0xDEAD, 1, step, env_idx)
    pods_bytes = []
    for i in range(4):
        team = i // 2
        p_thrust = float(actions[i, 0]) * 100.0
        p_reward = float(rewards[team]) if rewards is not None else 0.0
        p_data = struct.pack('<7f2H', 
            float(pos[i, 0]), float(pos[i, 1]),
            float(vel[i, 0]), float(vel[i, 1]),
            float(angle[i]),
            p_thrust,
            p_reward,
            int(laps[i]),
            int(next_cps[i])
        )
        pods_bytes.append(p_data)
    return header + b''.join(pods_bytes)

def parse_frame_mock(binary_data):
    # Mocking useTelemetry.ts logic
    view = memoryview(binary_data)
    offset = 0
    
    magic = struct.unpack_from('<H', view, offset)[0]
    step = struct.unpack_from('<I', view, offset + 3)[0]
    offset += 16
    
    pods = []
    for i in range(4):
        x, y, vx, vy, ang, thrust, rew = struct.unpack_from('<7f', view, offset)
        lap, ncp = struct.unpack_from('<2H', view, offset + 28)
        pods.append({'x': x, 'y': y, 'id': i})
        offset += 32
        
    return pods

def verify_flow():
    print("Initializing Flow Verification...")
    env = PodRacerEnv(num_envs=1, device='cpu')
    env.reset()
    
    # 1. Get Data from Env
    idx = 0
    pos = env.physics.pos[idx].detach().numpy()
    vel = env.physics.vel[idx].detach().numpy()
    angle = env.physics.angle[idx].detach().numpy()
    laps = env.laps[idx].detach().numpy()
    next_cps = env.next_cp_id[idx].detach().numpy()
    rewards = torch.zeros(2)
    actions = torch.zeros((4, 4))
    
    print(f"Env Pos (Pod 0): {pos[0]}")
    
    # 2. Pack
    binary = pack_frame_mock(pos, vel, angle, laps, next_cps, rewards, actions, 1, 0)
    
    # 3. Decode
    decoded_pods = parse_frame_mock(binary)
    
    print(f"Decoded Pos (Pod 0): [{decoded_pods[0]['x']}, {decoded_pods[0]['y']}]")
    
    # Check
    err_x = abs(pos[0][0] - decoded_pods[0]['x'])
    err_y = abs(pos[0][1] - decoded_pods[0]['y'])
    
    if err_x < 0.1 and err_y < 0.1:
        print("PASS: Coordinates matched.")
    else:
        print("FAIL: Coordinate mismatch!")

if __name__ == "__main__":
    verify_flow()
