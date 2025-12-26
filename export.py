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
    
    # 1. Fuse Enemy Encoder [0] (Linear 13->32)
    # W_new = W / std
    # b_new = b - W * mean / std
    
    layer = model.enemy_encoder[0]
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
    
    # 2. Fuse Backbone [0] (Linear 49->Hidden)
    # Structure: [Self(14), Teammate(13), Context(16), CP(6)]
    # Context(16) is internal, no fusion.
    # Teammate(13) uses rms_ent.
    
    layer = model.backbone[0]
    W = layer.weight.data.cpu().numpy() # [HD, 49]
    b = layer.bias.data.cpu().numpy()   # [HD]
    
    # Split W into sections
    W_self = W[:, 0:14]
    W_tm   = W[:, 14:27]
    W_ctx  = W[:, 27:43]
    W_cp   = W[:, 43:49]
    
    # Fuse Self
    W_self_new = W_self / std_s[None, :]
    shift_self = np.sum(W_self_new * mean_s[None, :], axis=1)

    # Fuse Teammate (Use std_e/mean_e)
    W_tm_new = W_tm / std_e[None, :]
    shift_tm = np.sum(W_tm_new * mean_e[None, :], axis=1)
    
    # Fuse CP
    W_cp_new = W_cp / std_c[None, :]
    shift_cp = np.sum(W_cp_new * mean_c[None, :], axis=1)
    
    # Reassemble
    W_new = np.concatenate([W_self_new, W_tm_new, W_ctx, W_cp_new], axis=1)
    b_new = b - shift_self - shift_tm - shift_cp
    
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
        model.enemy_encoder[0], # Linear 13->32
        model.enemy_encoder[2], # Linear 32->16
        model.backbone[0],       # Linear 49->HD
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

SINGLE_FILE_TEMPLATE = """import sys, math
W,H = {WIDTH},{HEIGHT}
SP = 1.0/{WIDTH}.0
SV = 1.0/1000.0
HD = {HIDDEN_DIM}

class N:
    def __init__(self,d,s):
        self.w = self.dec(d,s)
        self.c = 0
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
    def gw(self,n):
        r=self.w[self.c:self.c+n]; self.c+=n
        return r
    def lin(self,x,i,o,r=False):
        w,b,out=self.gw(i*o),self.gw(o),[]
        for k in range(o):
            a=b[k]
            for j in range(i): a+=x[j]*w[k*i+j]
            out.append(max(0.0,a) if r else a)
        return out

class A(N):
    def f(self,s,t,e,c):
        self.c=0
        w1,b1,w2,b2=self.gw(416),self.gw(32),self.gw(512),self.gw(16)
        encs=[]
        for en in e:
            h=[0.0]*32
            for r in range(32):
                a=b1[r]
                for j in range(13): a+=en[j]*w1[r*13+j]
                h[r]=max(0.0,a)
            z=[0.0]*16
            for r in range(16):
                a=b2[r]
                for j in range(32): a+=h[j]*w2[r*32+j]
                z[r]=a
            encs.append(z)
        g=[max(x[i] for x in encs) for i in range(16)] if encs else [0.0]*16
        x=s+t+g+c
        x=self.lin(x,49,HD,True)
        x=self.lin(x,HD,HD,True)
        th=self.lin(x,HD,1)[0]
        an=self.lin(x,HD,1)[0]
        sh=self.lin(x,HD,2)
        bo=self.lin(x,HD,2)
        th=1.0/(1.0+math.exp(-th))
        an=(math.exp(2*an)-1)/(math.exp(2*an)+1)
        return [th,an,1 if sh[1]>sh[0] else 0,1 if bo[1]>bo[0] else 0]

W_BLOB="{WEIGHTS_BLOB}"
SC={SCALE_VAL}

def tl(vx,vy,a):
    r=math.radians(a)
    c,s=math.cos(r),math.sin(r)
    return vx*c+vy*s, -vx*s+vy*c

def solve():
    m=A(W_BLOB,SC)
    try:
        input(); C=int(input())
        cps=[list(map(int,input().split())) for _ in range(C)]
        laps,p_ncp,to,scd,bavl=[0]*4,[1]*4,[100]*4,[0]*4,[True,True]
        while True:
            pods=[]
            for i in range(2): x,y,vx,vy,a,n=map(int,input().split()); pods.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':a,'n':n,'id':i,'tm':0})
            for i in range(2): x,y,vx,vy,a,n=map(int,input().split()); pods.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':a,'n':n,'id':i+2,'tm':1})
            for i in range(4):
                p=pods[i]
                if p['n']!=p_ncp[i]:
                    if p['n']==1 and p_ncp[i]==C: laps[i]+=1
                    elif p['n']==0 and p_ncp[i]==C-1: laps[i]+=1
                    to[i]=100
                else: to[i]-=1
                p_ncp[i]=p['n']
            scrs=[]
            for i in range(4):
                p=pods[i]; cp=cps[p['n']] if p['n']<len(cps) else cps[0]
                d=math.sqrt((p['x']-cp[0])**2+(p['y']-cp[1])**2)
                scrs.append(laps[i]*50000+p['n']*500+(20000-d))
            run=[False]*4
            if scrs[0]>=scrs[1]: run[0]=True
            else: run[1]=True
            
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
                fsh=scd[i]/3.0
                fbo=1.0 if bavl[0] else 0.0
                fto=to[i]/100.0
                fla=laps[i]/3.0
                fle=1.0 if run[i] else 0.0
                vm=math.sqrt(p['vx']**2+p['vy']**2)*SV
                oself=[fvf,fvr,ftf,ftr,ftd,fac,fas,fsh,fbo,fto,fla,fle,vm,0.0]
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
                    osh=1.0 if scd[j]>0 else 0.0
                    ot=cps[o['n']]
                    otx,oty=ot[0]-p['x'],ot[1]-p['y']
                    otf,otr=tl(otx,oty,p['a'])
                    feat=[dpf*SP,dpr*SP,dvf*SV,dvr*SV,math.cos(rr),math.sin(rr),dist,mate,osh,otf*SP,otr*SP,0.0,0.0]
                    if o['tm']==p['tm']: otm.extend(feat)
                    else: oen.append(feat)
                if not otm: otm=[0.0]*13
                c1=cps[p['n']]; c2=cps[(p['n']+1)%len(cps)]
                cx,cy=c2[0]-c1[0],c2[1]-c1[1]
                cf,cr=tl(cx,cy,p['a'])
                ocp=[ftf,ftr,cf*SP,cr*SP,0.0,0.0]
                out=m.f(oself,otm,oen,ocp)
                pw=str(int(out[0]*100))
                if out[2]==1 and scd[i]==0: pw="SHIELD"; scd[i]=4
                elif out[3]==1 and bavl[0]: pw="BOOST"; bavl[0]=False
                if scd[i]>0: scd[i]-=1
                tx=int(p['x']+math.cos(math.radians(p['a']+out[1]*18.0))*10000)
                ty=int(p['y']+math.sin(math.radians(p['a']+out[1]*18.0))*10000)
                print(f"{tx} {ty} {pw}")
            turn+=1
    except: pass   
if __name__=="__main__": solve()
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
