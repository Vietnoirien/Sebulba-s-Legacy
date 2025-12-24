import torch
import base64
import math
import argparse
import sys
import numpy as np
import os
from models.deepsets import PodAgent
from config import *

# Constants
MAX_CHARS = 100000

def fuse_normalization(model, rms_stats):
    """
    Fuses RunningMeanStd statistics into the first layer weights.
    model: PodAgent
    rms_stats: Dict {'self': state_dict, 'ent': state_dict, 'cp': state_dict}
    """
    print("Fusing Normalization Statistics...")
    
    # helper to get mean/std from state dict
    def get_ms(sd):
        mean = sd['mean'].cpu().numpy()
        var = sd['var'].cpu().numpy()
        std = np.sqrt(var + 1e-4) # epsilon match
        return mean, std
        
    mean_s, std_s = get_ms(rms_stats['self'])
    mean_e, std_e = get_ms(rms_stats['ent'])
    mean_c, std_c = get_ms(rms_stats['cp'])
    
    # 1. Fuse Entity Encoder [0] (Linear 13->32)
    # W_new = W / std
    # b_new = b - W * mean / std
    
    layer = model.entity_encoder[0]
    W = layer.weight.data.cpu().numpy() # [32, 13]
    b = layer.bias.data.cpu().numpy()   # [32]
    
    # Fuse
    # Broadcast std [13] to [32, 13]
    W_new = W / std_e[None, :]
    
    # Bias shift
    # sum(W_new * mean)
    bias_shift = np.sum(W_new * mean_e[None, :], axis=1)
    b_new = b - bias_shift
    
    # Update Layer
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)
    
    # 2. Fuse Backbone [0] (Linear 36->Hidden)
    # Structure: [Self(14), Context(16), CP(6)]
    # Context(16) is internal, no fusion.
    
    layer = model.backbone[0]
    W = layer.weight.data.cpu().numpy() # [HD, 36]
    b = layer.bias.data.cpu().numpy()   # [HD]
    
    # Split W into sections
    W_self = W[:, 0:14]
    W_ctx  = W[:, 14:30]
    W_cp   = W[:, 30:36]
    
    # Fuse Self
    W_self_new = W_self / std_s[None, :]
    shift_self = np.sum(W_self_new * mean_s[None, :], axis=1)
    
    # Fuse CP
    W_cp_new = W_cp / std_c[None, :]
    shift_cp = np.sum(W_cp_new * mean_c[None, :], axis=1)
    
    # Reassemble
    W_new = np.concatenate([W_self_new, W_ctx, W_cp_new], axis=1)
    b_new = b - shift_self - shift_cp
    
    # Update Layer
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)
    
    print("Fusion Complete.")

def quantize_weights(model):
    """
    Extracts weights from model, quantizes them to int8, and returns metadata.
    """
    weights = []
    
    ordered_layers = [
        model.entity_encoder[0], # Linear 13->32
        model.entity_encoder[2], # Linear 32->16
        model.backbone[0],       # Linear 36->HD
        model.backbone[2],       # Linear HD->HD
        model.actor_thrust_mean, # Linear HD->1
        model.actor_angle_mean,  # Linear HD->1
        model.actor_shield,      # Linear HD->2
        model.actor_boost        # Linear HD->2
    ]
    
    for layer in ordered_layers:
        # Weights: [Out, In]
        w = layer.weight.data.cpu().numpy().flatten()
        if layer.bias is not None:
            b = layer.bias.data.cpu().numpy().flatten()
        else:
            b = np.zeros(layer.out_features)
            
        weights.extend(w)
        weights.extend(b)
        
    print(f"Total Parameters: {len(weights)}")
    
    # Quantization
    # Simple MinMax scaling
    # Map [Min, Max] -> [-127, 127]
    min_val = min(weights)
    max_val = max(weights)
    scale = max(abs(min_val), abs(max_val)) / 127.0
    
    quantized = []
    for x in weights:
        q = int(round(x / scale))
        q = max(-127, min(127, q))
        quantized.append(q)
        
    return quantized, scale

def encode_data(quantized_data):
    # Base85 Encoding
    byte_valid = []
    for q in quantized_data:
        if q < 0:
            byte_valid.append(q + 256)
        else:
            byte_valid.append(q)
            
    # Pad to multiple of 4
    while len(byte_valid) % 4 != 0:
        byte_valid.append(0)
        
    encoded_str = ""
    # Process 4 bytes at a time
    for i in range(0, len(byte_valid), 4):
        chunk = byte_valid[i:i+4]
        # value = (b0 << 24) | (b1 << 16) | ...
        val = (chunk[0] << 24) | (chunk[1] << 16) | (chunk[2] << 8) | chunk[3]
        
        # Convert to base 85 (5 chars)
        chars = []
        for _ in range(5):
            chars.append(val % 85)
            val //= 85
        
        # Reverse because we want Big Endian for string?
        chars.reverse()
        
        for c in chars:
            encoded_str += chr(c + 33)
            
        # Line break every 100 chars? No, keep it compact.
            
    return encoded_str

SINGLE_FILE_TEMPLATE = """import sys
import math

# Constants
WIDTH = {WIDTH}
HEIGHT = {HEIGHT}
S_POS = 1.0 / {WIDTH}.0
S_VEL = 1.0 / 1000.0
HIDDEN_DIM = {HIDDEN_DIM}

class NN:
    def __init__(self, data_str, scale):
        self.weights = self.decode(data_str, scale)
        self.cursor = 0
    
    def decode(self, blob, display_scale):
        w = []
        val = 0
        count = 0
        for char in blob:
            val = val * 85 + (ord(char) - 33)
            count += 1
            if count == 5:
                for i in range(4):
                    b = (val >> (24 - i*8)) & 0xFF
                    w.append((b - 256 if b > 127 else b) * display_scale)
                val = 0
                count = 0
        return w

    def get_w(self, n):
        res = self.weights[self.cursor : self.cursor + n]
        self.cursor += n
        return res

    def linear_layer(self, x, in_d, out_d, relu=False):
        w = self.get_w(in_d * out_d)
        b = self.get_w(out_d)
        out = []
        for r in range(out_d):
            acc = b[r]
            for c in range(in_d):
                acc += x[c] * w[r*in_d + c]
            if relu: acc = max(0.0, acc)
            out.append(acc)
        return out

class Agent(NN):
    def forward(self, self_obs, entity_obs, cp_obs):
        self.cursor = 0
        
        # Shared Encoder Weights
        # L1: 13->32
        w1 = self.get_w(13*32); b1 = self.get_w(32)
        # L2: 32->16
        w2 = self.get_w(32*16); b2 = self.get_w(16)
        
        # Encode Entities
        encs = []
        for ent in entity_obs:
            # L1
            h = [0.0]*32
            for r in range(32):
                acc = b1[r]
                for c in range(13): acc += ent[c] * w1[r*13 + c]
                h[r] = max(0.0, acc)
            
            # L2
            z = [0.0]*16
            for r in range(16):
                acc = b2[r]
                for c in range(32): acc += h[c] * w2[r*32 + c]
                z[r] = acc 
            encs.append(z)
            
        # Max Pool
        g = [max(e[i] for e in encs) for i in range(16)]
        
        # Backbone
        # Input: Self(14) + Global(16) + CP(6) = 36
        x = self_obs + g + cp_obs 
        
        # L3: 36->HIDDEN_DIM
        x = self.linear_layer(x, 36, HIDDEN_DIM, relu=True)
        # L4: HIDDEN_DIM->HIDDEN_DIM
        x = self.linear_layer(x, HIDDEN_DIM, HIDDEN_DIM, relu=True)
        
        # Heads
        thrust = self.linear_layer(x, HIDDEN_DIM, 1, relu=False)[0]
        angle = self.linear_layer(x, HIDDEN_DIM, 1, relu=False)[0]
        shield = self.linear_layer(x, HIDDEN_DIM, 2, relu=False) # logits
        boost = self.linear_layer(x, HIDDEN_DIM, 2, relu=False) # logits
        
        # Activations
        # Thrust: Sigmoid -> 0..1
        thrust = 1.0 / (1.0 + math.exp(-thrust))
        # Angle: Tanh -> -1..1
        angle = (math.exp(2*angle) - 1) / (math.exp(2*angle) + 1)
        
        # Shield/Boost: Argmax
        shield_act = 1 if shield[1] > shield[0] else 0
        boost_act = 1 if boost[1] > boost[0] else 0
        
        return [thrust, angle, shield_act, boost_act]

# --- Game Loop ---
WEIGHTS = "{WEIGHTS_BLOB}"
SCALE = {SCALE_VAL}

def to_local(vx, vy, angle_deg):
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    rx = vx * c + vy * s
    ry = -vx * s + vy * c
    return rx, ry

def solve():
    model = Agent(WEIGHTS, SCALE)
    try:
        laps_total = int(input())
        checkpoint_count = int(input())
        checkpoints = []
        for _ in range(checkpoint_count):
            checkpoints.append(list(map(int, input().split())))
        laps = [0]*4
        prev_ncp = [1]*4 
        timeouts = [100]*4
        shield_cd = [0]*4
        boost_avail = [True, True] 
        while True:
            pods = []
            for i in range(2):
                 x, y, vx, vy, angle, next_cp = map(int, input().split())
                 pods.append({'x':x, 'y':y, 'vx':vx, 'vy':vy, 'angle':angle, 'ncp':next_cp, 'id':i, 'team':0})
            for i in range(2):
                 x, y, vx, vy, angle, next_cp = map(int, input().split())
                 pods.append({'x':x, 'y':y, 'vx':vx, 'vy':vy, 'angle':angle, 'ncp':next_cp, 'id':i+2, 'team':1})

            for i in range(4):
                p = pods[i]
                if p['ncp'] != prev_ncp[i]:
                    if p['ncp'] == 1 and prev_ncp[i] == checkpoint_count: laps[i] += 1
                    elif p['ncp'] == 0 and prev_ncp[i] == checkpoint_count - 1: laps[i] += 1
                    timeouts[i] = 100
                else: timeouts[i] -= 1
                prev_ncp[i] = p['ncp']
                
            scores = []
            for i in range(4):
                p = pods[i]
                cp_pos = checkpoints[p['ncp']] if p['ncp'] < len(checkpoints) else checkpoints[0]
                dist = math.sqrt((p['x']-cp_pos[0])**2 + (p['y']-cp_pos[1])**2)
                sc = laps[i] * 50000 + p['ncp'] * 500 + (20000 - dist)
                scores.append(sc)
            
            is_runner = [False]*4
            if scores[0] >= scores[1]: is_runner[0] = True
            else: is_runner[1] = True
            
            for i in range(2):
                p = pods[i]
                v_fwd, v_right = to_local(p['vx'], p['vy'], p['angle'])
                f_vfwd = v_fwd * S_VEL
                f_vright = v_right * S_VEL
                target = checkpoints[p['ncp']]
                gtx = target[0] - p['x']
                gty = target[1] - p['y']
                t_fwd, t_right = to_local(gtx, gty, p['angle'])
                f_tfwd = t_fwd * S_POS
                f_tright = t_right * S_POS
                f_tdist = math.sqrt(gtx**2 + gty**2) * S_POS
                dist_safe = f_tdist + 1e-6
                f_align_cos = (f_tfwd / S_POS) / (dist_safe / S_POS)
                f_align_sin = (f_tright / S_POS) / (dist_safe / S_POS)
                f_shield = shield_cd[i] / 3.0
                f_boost = 1.0 if boost_avail[0] else 0.0
                f_to = timeouts[i] / 100.0
                f_lap = laps[i] / 3.0
                f_leader = 1.0 if is_runner[i] else 0.0
                v_mag = math.sqrt(p['vx']**2 + p['vy']**2) * S_VEL
                obs_self = [f_vfwd, f_vright, f_tfwd, f_tright, f_tdist, f_align_cos, f_align_sin, f_shield, f_boost, f_to, f_lap, f_leader, v_mag, 0.0]
                
                obs_ents = []
                for j in range(4):
                    if i == j: continue
                    o = pods[j]
                    dx_g = o['x'] - p['x']
                    dy_g = o['y'] - p['y']
                    dvx_g = o['vx'] - p['vx']
                    dvy_g = o['vy'] - p['vy']
                    dp_fwd, dp_right = to_local(dx_g, dy_g, p['angle'])
                    dv_fwd, dv_right = to_local(dvx_g, dvy_g, p['angle'])
                    f_dp_fwd = dp_fwd * S_POS
                    f_dp_right = dp_right * S_POS
                    f_dv_fwd = dv_fwd * S_VEL
                    f_dv_right = dv_right * S_VEL
                    rel_angle = o['angle'] - p['angle']
                    rel_rad = math.radians(rel_angle)
                    f_cos = math.cos(rel_rad)
                    f_sin = math.sin(rel_rad)
                    dist = math.sqrt(dx_g**2 + dy_g**2) * S_POS
                    mate = 1.0 if (o['team'] == p['team']) else 0.0
                    o_shield = 1.0 if shield_cd[j] > 0 else 0.0
                    otarget = checkpoints[o['ncp']]
                    otx_g = otarget[0] - p['x']
                    oty_g = otarget[1] - p['y']
                    ot_fwd, ot_right = to_local(otx_g, oty_g, p['angle'])
                    f_ot_fwd = ot_fwd * S_POS
                    f_ot_right = ot_right * S_POS
                    feat = [f_dp_fwd, f_dp_right, f_dv_fwd, f_dv_right, f_cos, f_sin, dist, mate, o_shield, f_ot_fwd, f_ot_right, 0.0, 0.0]
                    obs_ents.append(feat)
                    
                cp1 = checkpoints[p['ncp']]
                cp2 = checkpoints[(p['ncp'] + 1) % len(checkpoints)]
                v12x_g = cp2[0] - cp1[0]
                v12y_g = cp2[1] - cp1[1]
                v12_fwd, v12_right = to_local(v12x_g, v12y_g, p['angle'])
                obs_cp = [f_tfwd, f_tright, v12_fwd * S_POS, v12_right * S_POS, 0.0, 0.0]
                
                outs = model.forward(obs_self, obs_ents, obs_cp)
                thrust_val = int(outs[0] * 100)
                angle_val = outs[1] * 18.0
                do_shield = (outs[2] == 1)
                do_boost = (outs[3] == 1)
                t_angle = p['angle'] + angle_val
                trad = math.radians(t_angle)
                tx = int(p['x'] + math.cos(trad) * 10000)
                ty = int(p['y'] + math.sin(trad) * 10000)
                power = str(thrust_val)
                if do_shield and shield_cd[i] == 0:
                    power = "SHIELD"
                    shield_cd[i] = 4
                elif do_boost and boost_avail[0]:
                    power = "BOOST"
                    boost_avail[0] = False
                if shield_cd[i] > 0: shield_cd[i] -= 1
                print(f"{tx} {ty} {power}")
            turn += 1
    except Exception: pass

if __name__ == "__main__":
    solve()
"""

def export_model(model_path, output_path="submission.py"):
    # Load Model
    agent = PodAgent()
    try:
        agent.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading simple state dict if Wrapper present
        ckpt = torch.load(model_path, map_location='cpu')
        if 'state_dict' in ckpt:
            agent.load_state_dict(ckpt['state_dict'])
        else:
            raise e
            
    # Try Loading Normalization Stats
    model_dir = os.path.dirname(model_path)
    # Assuming standard structure data/generations/gen_X/agent_Y.pt
    # Stats are in data/generations/gen_X/rms_stats.pt
    rms_path = os.path.join(model_dir, "rms_stats.pt")
    
    if os.path.exists(rms_path):
        print(f"Loading RMS stats from {rms_path}")
        rms_stats = torch.load(rms_path, map_location='cpu')
        fuse_normalization(agent, rms_stats)
    else:
        print("WARNING: No RMS stats found! Model export will lack normalization.")
            
    # Quantize
    q_data, scale = quantize_weights(agent)
    encoded = encode_data(q_data)
    
    print(f"Encoded Size: {len(encoded)} chars")
    
    # Escape
    encoded_escaped = encoded.replace("\\", "\\\\").replace("\"", "\\\"")
    
    hidden_dim = getattr(agent, 'hidden_dim', 256) # Default to 256 now
    
    script = SINGLE_FILE_TEMPLATE.replace("{WEIGHTS_BLOB}", encoded_escaped)\
        .replace("{SCALE_VAL}", str(scale))\
        .replace("{WIDTH}", str(WIDTH))\
        .replace("{HEIGHT}", str(HEIGHT))\
        .replace("{HIDDEN_DIM}", str(hidden_dim))
    
    with open(output_path, 'w') as f:
        f.write(script)
        
    print(f"Exported to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--out", type=str, default="submission.py")
    args = parser.parse_args()
    
    export_model(args.model, args.out)

if __name__ == "__main__":
    main()
