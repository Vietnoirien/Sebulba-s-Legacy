import torch
import base64
import math
import argparse
import sys
import numpy as np
import os
import json
from models.deepsets import PodAgent
from config import *

# Constants
MAX_CHARS = 100000

def get_ms(sd):
    mean = sd['mean'].cpu().numpy()
    var = sd['var'].cpu().numpy()
    std = np.sqrt(var + 1e-4)
    return mean, std

def fuse_layer_section(W, mean, std, threshold=0.05):
    W_new = np.zeros_like(W)
    safe_mask = std > threshold
    W_new[:, safe_mask] = W[:, safe_mask] / std[None, safe_mask]
    shift = np.sum(W_new * mean[None, :], axis=1)
    return W_new, shift

def fuse_normalization_pilot(pilot, rms_stats):
    """
    Fuses RMS stats into Pilot Embed.
    Input: Self(15) + CP(6) = 21
    Target: pilot.pilot_embed[0] (Linear 21->64)
    """
    print("Fusing Pilot Normalization...")
    mean_s, std_s = get_ms(rms_stats['self'])
    mean_c, std_c = get_ms(rms_stats['cp'])
    
    layer = pilot.pilot_embed[0]
    W = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()
    
    # Split W: [64, 21] -> [64, 15], [64, 6]
    W_self = W[:, 0:15]
    W_cp   = W[:, 15:21]
    
    W_self_new, shift_self = fuse_layer_section(W_self, mean_s, std_s)
    W_cp_new, shift_cp = fuse_layer_section(W_cp, mean_c, std_c)
    
    W_new = np.concatenate([W_self_new, W_cp_new], axis=1)
    b_new = b - shift_self - shift_cp
    
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)

def fuse_normalization_commander(actor, rms_stats):
    """
    Fuses RMS stats into Commander components.
    1. Enemy Encoder (Linear 13->32) -> RMS(Ent)
    2. Commander Backbone (Linear 63->128) -> RMS(Self)
    """
    print("Fusing Commander Normalization...")
    mean_s, std_s = get_ms(rms_stats['self'])
    mean_e, std_e = get_ms(rms_stats['ent'])
    
    # 1. Enemy Encoder [0] (13 -> 32)
    layer = actor.enemy_encoder[0]
    W = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()
    
    W_new, shift = fuse_layer_section(W, mean_e, std_e)
    b_new = b - shift
    
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)
    
    # 2. Backbone [0] (63 -> 128)
    # Input: Self(15) + Team(16) + Ctx(16) + Role(16)
    layer = actor.commander_backbone[0]
    W = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()
    
    W_self = W[:, 0:15]
    W_rest = W[:, 15:]
    
    W_self_new, shift_self = fuse_layer_section(W_self, mean_s, std_s)
    
    W_new = np.concatenate([W_self_new, W_rest], axis=1)
    b_new = b - shift_self
    
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)

def extract_layer(layer, bias=True):
    w = layer.weight.data.cpu().numpy().flatten()
    if bias and layer.bias is not None:
        b = layer.bias.data.cpu().numpy().flatten()
        return np.concatenate([w, b])
    return w

def quantize_weights(actor):
    """
    Traverses the New Layout and extracts weights.
    Structure:
     1. Pilot Embed (2 layers)
     2. Enemy Enc (2 layers)
     3. Role Emb (No bias)
     4. Cmd Backbone (2 layers)
     5. LSTM (Wi, Wh, bi, bh)
     6. Heads (4 layers)
    """
    weights = []
    
    # 1. Pilot Embed
    # [0], [2]
    weights.extend(extract_layer(actor.pilot_embed[0]))
    weights.extend(extract_layer(actor.pilot_embed[2]))
    
    # 2. Enemy Enc
    # [0], [2]
    weights.extend(extract_layer(actor.enemy_encoder[0]))
    weights.extend(extract_layer(actor.enemy_encoder[2]))
    
    # 3. Role Emb
    weights.extend(actor.role_embedding.weight.data.cpu().numpy().flatten())
    
    # 4. Cmd Backbone
    # [0], [2]
    weights.extend(extract_layer(actor.commander_backbone[0]))
    weights.extend(extract_layer(actor.commander_backbone[2]))
    
    # 5. LSTM
    # input_size=128, hidden=64.
    # We want: Wi(256, 128), Wh(256, 64), B(256)
    lstm = actor.lstm
    # PyTorch stores: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    # CustomLSTM stores: ih.weight, hh.weight, ih.bias, hh.bias
    if hasattr(lstm, 'weight_ih_l0'): # Legacy nn.LSTM support
        wi = lstm.weight_ih_l0.data.cpu().numpy().flatten()
        wh = lstm.weight_hh_l0.data.cpu().numpy().flatten()
        bi = lstm.bias_ih_l0.data.cpu().numpy()
        bh = lstm.bias_hh_l0.data.cpu().numpy()
    else: # CustomLSTM
        wi = lstm.ih.weight.data.cpu().numpy().flatten()
        wh = lstm.hh.weight.data.cpu().numpy().flatten()
        bi = lstm.ih.bias.data.cpu().numpy()
        bh = lstm.hh.bias.data.cpu().numpy()
        
    b = bi + bh
    
    weights.extend(wi)
    weights.extend(wh)
    weights.extend(b)
    
    # 6. Heads
    weights.extend(extract_layer(actor.head_thrust))
    weights.extend(extract_layer(actor.head_angle))
    weights.extend(extract_layer(actor.head_shield))
    weights.extend(extract_layer(actor.head_boost))

    # 7. Map Encoder (Transformer)
    # 2 -> 32 (Linear)
    weights.extend(extract_layer(actor.map_encoder.embedding))
    
    # Layer 0 Only (Fixed 1 layer)
    # Self Attn: in_proj_weight, in_proj_bias (3 * 32 -> 96)
    # out_proj: weight, bias (32 -> 32)
    # Norm1: weight, bias (32)
    # Lin1: weight, bias (32 -> 64) (d_model -> dim_feedforward)
    # Lin2: weight, bias (64 -> 32)
    # Norm2: weight, bias (32)
    
    enc = actor.map_encoder.transformer_encoder.layers[0]
    
    # In Proj
    weights.extend(enc.self_attn.in_proj_weight.data.cpu().numpy().flatten())
    weights.extend(enc.self_attn.in_proj_bias.data.cpu().numpy().flatten())
    
    # Out Proj (Non-Linear Layer in Torch, but Linear in Logic)
    weights.extend(enc.self_attn.out_proj.weight.data.cpu().numpy().flatten())
    weights.extend(enc.self_attn.out_proj.bias.data.cpu().numpy().flatten())
    
    # Norm1
    weights.extend(enc.norm1.weight.data.cpu().numpy().flatten())
    weights.extend(enc.norm1.bias.data.cpu().numpy().flatten())
    
    # FF Lin1
    weights.extend(enc.linear1.weight.data.cpu().numpy().flatten())
    weights.extend(enc.linear1.bias.data.cpu().numpy().flatten())
    
    # FF Lin2
    weights.extend(enc.linear2.weight.data.cpu().numpy().flatten())
    weights.extend(enc.linear2.bias.data.cpu().numpy().flatten())
    
    # Norm2
    weights.extend(enc.norm2.weight.data.cpu().numpy().flatten())
    weights.extend(enc.norm2.bias.data.cpu().numpy().flatten())
    
    # Quantize
    min_val, max_val = min(weights), max(weights)
    scale = max(abs(min_val), abs(max_val)) / 127.0
    
    quantized = []
    for x in weights:
        q = int(round(x / scale))
        q = max(-127, min(127, q))
        quantized.append(q)
        
    return quantized, scale

def encode_data(quantized_data):
    byte_valid = []
    for q in quantized_data:
        if q < 0: byte_valid.append(q + 256)
        else: byte_valid.append(q)
    while len(byte_valid) % 4 != 0:
        byte_valid.append(0)
    encoded_str = ""
    for i in range(0, len(byte_valid), 4):
        chunk = byte_valid[i:i+4]
        val = (chunk[0] << 24) | (chunk[1] << 16) | (chunk[2] << 8) | chunk[3]
        chars = []
        for _ in range(5):
            chars.append(val % 85)
            val //= 85
        chars.reverse()
        for c in chars: encoded_str += chr(c + 33)
    return encoded_str

def minify_code(code):
    import re
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\n\s*\n', '\n', code)
    code = code.strip()
    return code

FAST_PHYSICS_CODE = """
import math
def clip(v, mn, mx): return max(mn, min(mx, v))
def sim(pods, cps, acts):
    for i, p in enumerate(pods):
        ua = acts[i][1]
        da = clip(ua, -18.0, 18.0)
        p['a'] += da
        if p['a'] > 180: p['a'] -= 360
        elif p['a'] <= -180: p['a'] += 360
        th = clip(acts[i][0], 0, 100)
        if acts[i][2]: 
             p['s'] = 4; p['m'] = 10.0; th = 0
        else:
             if p['s'] > 0: p['s'] -= 1; th = 0
             p['m'] = 10.0 if p['s'] == 3 else 1.0
        r = math.radians(p['a'])
        p['vx'] += th * math.cos(r); p['vy'] += th * math.sin(r)
        p['x'] += p['vx']; p['y'] += p['vy']
    for _ in range(2): 
        pairs = []
        for i in range(4):
            for j in range(i+1, 4):
                 dx = pods[j]['x'] - pods[i]['x']; dy = pods[j]['y'] - pods[i]['y']
                 d2 = dx*dx + dy*dy; rs = 800.0
                 if d2 < rs*rs:
                     d = math.sqrt(d2); d = 1e-4 if d < 1e-4 else d
                     nx, ny = dx/d, dy/d
                     dvx = pods[i]['vx'] - pods[j]['vx']; dvy = pods[i]['vy'] - pods[j]['vy']
                     prod = dvx*nx + dvy*ny
                     m1 = pods[i].get('m', 1.0); m2 = pods[j].get('m', 1.0)
                     f = prod / (1/m1 + 1/m2)
                     if f < 120.0 and prod > 0: f = 120.0
                     elif prod <= 0: f = 0
                     jx, jy = -f*nx, -f*ny
                     ov = rs - d; sep = ov / 2.0; sx, sy = nx*sep, ny*sep
                     pairs.append((i, j, jx, jy, sx, sy, m1, m2))
        for i, j, jx, jy, sx, sy, m1, m2 in pairs:
             pods[i]['vx'] += jx/m1; pods[i]['vy'] += jy/m1; pods[i]['x'] -= sx; pods[i]['y'] -= sy
             pods[j]['vx'] -= jx/m2; pods[j]['vy'] -= jy/m2; pods[j]['x'] += sx; pods[j]['y'] += sy
    for p in pods:
        p['vx'] *= 0.85; p['vy'] *= 0.85; p['x'] = round(p['x']); p['y'] = round(p['y']); p['vx'] = int(p['vx']); p['vy'] = int(p['vy'])
        cp = cps[p['n']]
        if (p['x']-cp[0])**2 + (p['y']-cp[1])**2 < 360000:
            p['n'] = (p['n'] + 1)
            if p['n'] >= len(cps): p['n'] = 0; p['laps'] += 1
            p['to'] = 100
        else: p['to'] -= 1
    return pods
"""

# --- UPDATED TEMPLATE ---
SINGLE_FILE_TEMPLATE = """import sys, math
SP=1.0/{WIDTH}.0; SV=1.0/1000.0

# --- FAST PHYSICS ---
{FAST_PHYSICS}
# --------------------

class N:
    def __init__(self,d,s):
        self.w=self.dec(d,s); self.c=0
        self.h=[0.0]*{LSTM}; self.C=[0.0]*{LSTM} # LSTM State
    def dec(self,b,s):
        w,v,cnt=[],0,0
        for c in b:
            v=v*85+(ord(c)-33); cnt+=1
            if cnt==5:
                for i in range(4):
                    x=(v>>(24-i*8))&0xFF
                    w.append((x-256 if x>127 else x)*s)
                v,cnt=0,0
        return w
    def gw(self,n): r=self.w[self.c:self.c+n]; self.c+=n; return r
    def lin(self,x,i,o,bias=True,relu=False):
        w,b,out=self.gw(i*o), self.gw(o) if bias else [0.0]*o, []
        for k in range(o):
            a=b[k]
            for j in range(i): a+=x[j]*w[k*i+j]
            out.append(max(0.0,a) if relu else a)
        return out
    def emb(self, idx, dim):
        w = self.gw(2 * dim)
        s = int(idx) * dim
        return w[s : s+dim]
    def lstm(self, x):
        # x(128) -> h({LSTM}), c({LSTM})
        # Weights: Wi({GATES}, 128), Wh({GATES}, {LSTM}), B({GATES})
        # Flattened gates: i, f, g, o
        wi = self.lin(x, 128, {GATES}, bias=False, relu=False) # Consumes Wi
        wh = self.lin(self.h, {LSTM}, {GATES}, bias=False, relu=False) # Consumes Wh
        b = self.gw({GATES})
        
        nc, nh = [], []
        for k in range({LSTM}):
            # Extract gate components
            # Offset k for each gate block (0..LSTM-1)
            # Order: i, f, g, o
            
            idx_i, idx_f, idx_g, idx_o = k, {LSTM}+k, 2*{LSTM}+k, 3*{LSTM}+k
            
            gate_i = wi[idx_i] + wh[idx_i] + b[idx_i]
            gate_f = wi[idx_f] + wh[idx_f] + b[idx_f]
            gate_g = wi[idx_g] + wh[idx_g] + b[idx_g]
            gate_o = wi[idx_o] + wh[idx_o] + b[idx_o]
            
            sig_i = 1.0/(1.0+math.exp(-gate_i))
            sig_f = 1.0/(1.0+math.exp(-gate_f))
            sig_o = 1.0/(1.0+math.exp(-gate_o))
            tanh_g = math.tanh(gate_g)
            
            new_c = sig_f * self.C[k] + sig_i * tanh_g
            new_h = sig_o * math.tanh(new_c)
            
            nc.append(new_c); nh.append(new_h)
            
        self.C, self.h = nc, nh
        return nh

        self.C, self.h = nc, nh
        return nh
    
    def map_tr(self, m):
        # m: List of [x, y] checks (MaxCP)
        # 1. Embed: 2 -> 32
        # Start of Map Weights.
        # Need to track offset manually or cleaner way?
        # Let's count bytes consumed by previous layers.
        # Pilot(21->64->64)= 1344 + 4096 = 5440 + biases...
        # Dynamic 'c' handles offset.
        
        # Linear Embed (2->32)
        w_emb = self.gw(2*32); b_emb = self.gw(32)
        
        # Transformer Weights
        # In Proj (32 -> 96)
        w_in = self.gw(32*96); b_in = self.gw(96)
        # Out Proj (32 -> 32)
        w_out = self.gw(32*32); b_out = self.gw(32)
        # Norm1 (32)
        g_n1 = self.gw(32); b_n1 = self.gw(32)
        # Lin1 (32 -> 64)
        w_l1 = self.gw(32*64); b_l1 = self.gw(64)
        # Lin2 (64 -> 32)
        w_l2 = self.gw(64*32); b_l2 = self.gw(32)
        # Norm2 (32)
        g_n2 = self.gw(32); b_n2 = self.gw(32)
        
        # Helper: Linear
        def L(x, w, b, i, o):
            res = []
            for k in range(o):
                acc = b[k]
                for j in range(i): acc += x[j]*w[k*i+j]
                res.append(acc)
            return res
            
        # Helper: Norm
        def N(x, g, b):
            mu = sum(x)/len(x)
            var = sum((v-mu)**2 for v in x)/len(x)
            std = math.sqrt(var + 1e-5)
            return [(x[i]-mu)/std * g[i] + b[i] for i in range(len(x))]

        # Execution
        # 1. Embedding
        seq = []
        for checks in m: # checks is [x, y]
            seq.append(L(checks, w_emb, b_emb, 2, 32))
            
        # 2. Transformer Layer (1 Layer)
        # Self Attention
        # Single Head logic since nhead=2 is complex to golf? 
        # No, must implement MHA. D=32, H=2, Dh=16.
        # Input: seq [N, 32]
        
        skip = [s[:] for s in seq] # Residual
        
        # QKV
        # In_proj weights are packed [3*D, D] -> [96, 32]
        # output is [N, 96] -> split into Q, K, V
        qkv_seq = [L(s, w_in, b_in, 32, 96) for s in seq]
        
        atten_out = []
        scale = 1.0 / 4.0 # sqrt(16)
        
        for t in range(len(seq)):
            # Per Head
            row_out = [0.0]*32
            for h in range(2):
                # Extract Q for this time step and head
                # Q start 0, K start 32, V start 64
                # Head 0: 0-15. Head 1: 16-31.
                qs = 0 + h*16; ke = 32 + h*16; ve = 64 + h*16
                
                # My Q
                q_vec = qkv_seq[t][qs : qs+16]
                
                # Attention Scores
                scores = []
                for s_idx in range(len(seq)):
                    k_vec = qkv_seq[s_idx][ke : ke+16]
                    dot = sum(q_vec[z]*k_vec[z] for z in range(16))
                    scores.append(dot * scale)
                
                # Softmax
                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                sum_e = sum(exps)
                probs = [e/sum_e for e in exps]
                
                # Weighted Sum V
                head_v = [0.0]*16
                for s_idx, prob in enumerate(probs):
                    v_vec = qkv_seq[s_idx][ve : ve+16]
                    for z in range(16): head_v[z] += prob * v_vec[z]
                    
                # Concatenate back to row
                for z in range(16): row_out[h*16 + z] = head_v[z]
                
            atten_out.append(row_out)
            
        # Out Proj
        proj_out = [L(a, w_out, b_out, 32, 32) for a in atten_out]
        
        # Add & Norm
        norm1_out = []
        for i in range(len(seq)):
            summed = [skip[i][j] + proj_out[i][j] for j in range(32)]
            norm1_out.append(N(summed, g_n1, b_n1))
            
        skip2 = [x[:] for x in norm1_out]
        
        # FF
        ff_out = []
        for x in norm1_out:
            # Linear1 + ReLU
            l1 = L(x, w_l1, b_l1, 32, 64)
            l1 = [max(0.0, v) for v in l1]
            # Linear2
            l2 = L(l1, w_l2, b_l2, 64, 32)
            ff_out.append(l2)
            
        # Add & Norm 2 (Output)
        final_seq = []
        for i in range(len(seq)):
            summed = [skip2[i][j] + ff_out[i][j] for j in range(32)]
            final_seq.append(N(summed, g_n2, b_n2))
            
        # Max Pool over sequence
        pooled = [max(chk[j] for chk in final_seq) for j in range(32)]
        return pooled

blob="{BLOB_UNIV}"; scale={SCALE_VAL}

def tl(vx,vy,a):
    r=math.radians(a)
    c,s=math.cos(r),math.sin(r)
    return vx*c+vy*s, -vx*s+vy*c

def solve():
    mr=B(blob,scale)
    if sys.version_info.minor >= 0: 
        try: input() 
        except: pass
    try: C=int(input())
    except: C=3
    cps=[list(map(int,input().split())) for _ in range(C)]
    laps,p_ncp,to,scd,bavl=[0]*4,[1]*4,[100]*4,[0]*4,[True,True]
    
    while True:
        pods=[]
        try:
            line = input()
            if not line: break
            parts = list(map(int,line.split()))
            x,y,vx,vy,a,n = parts
            pods.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':a,'n':n,'id':0,'tm':0,'s':scd[0],'m':10.0 if scd[0]==3 else 1.0,'to':to[0],'laps':laps[0]})
        except EOFError: break
            
        for i in range(1,4):
             x,y,vx,vy,a,n=map(int,input().split())
             tm = 0 if i < 2 else 1
             s_val = scd[i] if i < 2 else 0
             pods.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':a,'n':n,'id':i,'tm':tm,'s':s_val,'m':1.0,'to':to[i],'laps':laps[i]})
        
        for i in range(4):
            p=pods[i]
            if p['n']!=p_ncp[i]:
                if p['n']==1 and p_ncp[i]==0: laps[i]+=1
                to[i]=100
            else: to[i]-=1
            p_ncp[i]=p['n']; pods[i]['laps']=laps[i]; pods[i]['to']=to[i]
            
        scrs=[]
        for i in range(4):
            p=pods[i]; cp=cps[p['n']] if p['n']<len(cps) else cps[0]
            d=math.sqrt((p['x']-cp[0])**2+(p['y']-cp[1])**2)
            eff_cp = C if p['n'] == 0 else p['n']
            scrs.append(laps[i]*50000+eff_cp*500+(20000-d)/20)
            
        run=[False]*4
        if scrs[0]>=scrs[1]: run[0]=True
        else: run[1]=True
        if scrs[2]>=scrs[3]: run[2]=True
        else: run[3]=True
        
        for i in range(2):
            p=pods[i]
            vf,vr=tl(p['vx'],p['vy'],p['a'])
            fvf,fvr=vf*SV,vr*SV
            tar=cps[p['n']]
            gx,gy=tar[0]-p['x'],tar[1]-p['y']
            tf,tr=tl(gx,gy,p['a'])
            ftf,ftr=tf*SP,tr*SP
            ftd=math.sqrt(gx**2+gy**2)*SP
            ds=ftd+1e-6
            fac,fas=(ftf/SP)/(ds/SP),(ftr/SP)/(ds/SP)
            fsh=scd[i]/3.0; fbo=1.0 if bavl[0] else 0.0
            fto=to[i]/100.0; fla=laps[i]/3.0; fle=1.0 if run[i] else 0.0
            vm=math.sqrt(p['vx']**2+p['vy']**2)*SV
            rnk=0
            for s in scrs:
                if s>scrs[i]: rnk+=1
            flr=rnk/3.0
            oself=[fvf,fvr,ftf,ftr,ftd,fac,fas,fsh,fbo,fto,fla,fle,vm,0.0,flr]
            
            otm,oen=[],[]
            for j in range(4):
                if i==j: continue
                o=pods[j]
                dx,dy=o['x']-p['x'],o['y']-p['y']
                dvx,dvy=o['vx']-p['vx'],o['vy']-p['vy']
                dpf,dpr=tl(dx,dy,p['a'])
                dvf,dvr=tl(dvx,dvy,p['a'])
                ra=o['a']-p['a']; rr=math.radians(ra)
                dist=math.sqrt(dx**2+dy**2)*SP
                mate=1.0 if o['tm']==p['tm'] else 0.0
                osh=0.0
                ot=cps[o['n']]
                otx,oty=ot[0]-p['x'],ot[1]-p['y']
                otf,otr=tl(otx,oty,p['a'])
                o_run=1.0 if run[j] else 0.0
                o_rnk=0
                for s in scrs:
                    if s>scrs[j]: o_rnk+=1
                o_flr=o_rnk/3.0
                feat=[dpf*SP,dpr*SP,dvf*SV,dvr*SV,math.cos(rr),math.sin(rr),dist,mate,osh,otf*SP,otr*SP,o_run,o_flr]
                if o['tm']==p['tm']: otm.extend(feat)
                else: oen.append(feat)
            if not otm: otm=[0.0]*13
            c1=cps[p['n']]; c2=cps[(p['n']+1)%len(cps)]
            cx,cy=c2[0]-c1[0],c2[1]-c1[1]
            cf,cr=tl(cx,cy,p['a'])
            o_map = []
            # Relative Map Coords
            # N checks = len(cps). Input is [x, y].
            # Order: From Next CP (p['n']) to end, then start to p['n']-1 (Canonical)
            cnt = len(cps)
            start_n = p['n']
            for k in range(cnt):
                 idx = (start_n + k) % cnt
                 cx, cy = cps[idx][0] - p['x'], cps[idx][1] - p['y']
                 cf, cr = tl(cx, cy, p['a'])
                 o_map.append([cf*SP, cr*SP])
            
            # Use Single Brain with Role Input
            # Reset Check: If New Episode or Respawn or First Step
            # For submission, we can't easily detect Episode change except external var.
            # But the loop runs continuously.
            # "Game Loop" implies persistence.
            # We should reset if 'to' is high? or just never reset.
            # Never reset is safer for continuity.
            
            role_val = 1.0 if run[i] else 0.0
            out = mr.f(oself,otm,oen,ocp, o_map, role_val, False)
            
            rl_th = int(out[0]*100)
            rl_ang = out[1]*18.0
            rl_sh = (out[2]==1)
            rl_bo = (out[3]==1)
            
            current_abs_angle = pods[i]['a']
            rl_abs_angle = current_abs_angle + rl_ang
            
            best_act = (rl_th, rl_abs_angle, rl_sh, False) 
            
            vars = [0, 3, -3, 6, -6, 12, -12]
            candidates = []
            candidates.append((rl_th, rl_abs_angle, rl_sh))
            
            if not rl_sh:
                for da in vars:
                    candidates.append((100, rl_abs_angle + da, False)) 
                candidates.append((0, rl_abs_angle, False))
            
            best_score = -999999.0
            
            for (th, ang, sh) in candidates:
                sim_pods = [p.copy() for p in pods]
                acts = [[0,0,0]] * 4
                acts[i] = [th, ang - sim_pods[i]['a'], sh]
                
                final_pods = sim(sim_pods, cps, acts)
                fp = final_pods[i]
                
                score = 0
                cp_idx = fp['n']
                cp_pos = cps[cp_idx]
                dist = math.sqrt((fp['x']-cp_pos[0])**2 + (fp['y']-cp_pos[1])**2)
                
                score += (fp['n'] + fp['laps']*len(cps)) * 50000 - dist
                
                if (run[i]):
                     dx, dy = cp_pos[0]-fp['x'], cp_pos[1]-fp['y']
                     t_ang = math.degrees(math.atan2(dy, dx))
                     d_a = abs(t_ang - fp['a'])
                     while d_a > 180: d_a = 360 - d_a
                     score -= d_a * 10
                else:
                     d_a_rl = abs(ang - rl_ang)
                     score -= d_a_rl * 50
                
                if fp['x'] < 400 or fp['x'] > 15600 or fp['y'] < 400 or fp['y'] > 8600:
                    score -= 100000 
                
                if score > best_score:
                    best_score = score
                    best_act = (th, ang, sh, rl_bo)
            
            f_th, f_ang, f_sh, f_bo = best_act
            
            tx = int(p['x'] + 10000 * math.cos(math.radians(f_ang)))
            ty = int(p['y'] + 10000 * math.sin(math.radians(f_ang)))
            
            pw = str(int(f_th))
            if f_sh and scd[i] == 0: pw="SHIELD"; scd[i]=4
            elif f_bo and bavl[0] and not f_sh: pw="BOOST"; bavl[0]=False
            
            if scd[i]>0: scd[i]-=1
            print(f"{tx} {ty} {pw}")
            
if __name__=="__main__": solve()
"""

# Re-include helper functions (find_best_checkpoint) unchanged or adjusted to not depend on HP/HC
def find_best_checkpoint():
    league_path = "data/league.json"
    if not os.path.exists(league_path):
        print(f"Error: {league_path} not found. Cannot use --auto.")
        sys.exit(1)
        
    try:
        with open(league_path, "r") as f:
            registry = json.load(f)
    except Exception as e:
        print(f"Error reading league.json: {e}")
        sys.exit(1)
        
    if not registry:
        print("Error: League registry is empty.")
        sys.exit(1)
        
    valid_entries = [e for e in registry if "step" in e and "metrics" in e and "wins_ema" in e["metrics"]]
    
    if not valid_entries:
        print("Error: No valid entries with metrics found in league.")
        sys.exit(1)
        
    max_step = max(e["step"] for e in valid_entries)
    latest_gen = [e for e in valid_entries if e["step"] == max_step]
    latest_gen.sort(key=lambda x: x["metrics"]["wins_ema"], reverse=True)
    
    best_agent = latest_gen[0]
    print(f"Auto-selected Best Agent: {best_agent['id']} | Gen: {max_step} | Win Rate: {best_agent['metrics']['wins_ema']:.4f}")
    
    if "data/checkpoints/" in best_agent["path"]:
        filename = os.path.basename(best_agent["path"])
        parts = filename.replace(".pt", "").split("_")
        if len(parts) >= 4 and parts[0] == "gen" and parts[2] == "agent":
            gen_id = parts[1]
            agent_id = parts[3]
            base_data = "data"
            for stage in os.listdir(base_data):
                if stage.startswith("stage_"):
                    potential_path = os.path.join(base_data, stage, f"gen_{gen_id}", f"agent_{agent_id}.pt")
                    if os.path.exists(potential_path):
                        print(f"Resolved original path: {potential_path} (Contains RMS stats)")
                        return potential_path
                        
    return best_agent["path"]

def export_model(model_path, output_path="submission.py"):
    # Load Model
    agent = PodAgent() 
    try:
        agent.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f"Error loading model: {e}")
        ckpt = torch.load(model_path, map_location='cpu')
        if 'state_dict' in ckpt:
            agent.load_state_dict(ckpt['state_dict'])
        else:
            raise e
            
    # Normalize
    model_dir = os.path.dirname(model_path)
    rms_path = os.path.join(model_dir, "rms_stats.pt")
    
    if os.path.exists(rms_path):
        print(f"Loading RMS stats from {rms_path}")
        rms_stats = torch.load(rms_path, map_location='cpu')
        
        fuse_normalization_pilot(agent.actor, rms_stats)
        fuse_normalization_commander(agent.actor, rms_stats)
    else:
        print("WARNING: No RMS stats found!")
            
    # Quantize Universal
    q_univ, scale_univ = quantize_weights(agent.actor)
    
    enc_univ = encode_data(q_univ)
    
    # Escape
    univ_esc = enc_univ.replace("\\", "\\\\").replace("\"", "\\\"")
    
    # Minify Physics
    # Import FAST_PHYSICS_CODE from where? It was inline string in original file.
    # Re-declare it.
    fast_phys = minify_code(FAST_PHYSICS_CODE)
    
    
    # Retrieve Dims
    hp = agent.actor.pilot_embed[0].out_features # 64
    hc = agent.actor.commander_backbone[2].out_features # 64
    lstm_h = agent.actor.lstm.hidden_size # 48
    gates = 4 * lstm_h # 192

    script = SINGLE_FILE_TEMPLATE.replace("{BLOB_UNIV}", univ_esc)\
        .replace("{SCALE_VAL}", str(scale_univ))\
        .replace("{WIDTH}", str(WIDTH))\
        .replace("{HEIGHT}", str(HEIGHT))\
        .replace("{FAST_PHYSICS}", fast_phys)\
        .replace("{LSTM}", str(lstm_h))\
        .replace("{GATES}", str(gates))
    
    with open(output_path, 'w') as f:
        f.write(script)
        
    print(f"Exported to {output_path}")
    print(f"Total Chars: {len(script)}")
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to .pt model")
    parser.add_argument("--out", type=str, default="submission.py")
    parser.add_argument("--auto", action="store_true", help="Automatically select best checkpoint from league.json")
    args = parser.parse_args()
    
    if args.auto:
        model_path = find_best_checkpoint()
    elif args.model:
        model_path = args.model
    else:
        print("Error: Must specify --model or --auto")
        sys.exit(1)
    
    export_model(model_path, args.out)

if __name__ == "__main__":
    main()
