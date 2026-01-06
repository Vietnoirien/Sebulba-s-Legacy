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

def fuse_normalization_pilot(model, rms_stats):
    """
    Fuses RMS stats into PilotNet.
    Input: Self(15) + CP(6) = 21
    """
    print("Fusing Pilot Normalization...")
    # Get Stats
    mean_s, std_s = get_ms(rms_stats['self'])
    mean_c, std_c = get_ms(rms_stats['cp'])
    
    # Layer 0: Linear(21 -> 32)
    layer = model.net[0]
    W = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()
    
    # Split W: [32, 21] -> [32, 15] (Self), [32, 6] (CP)
    W_self = W[:, 0:15]
    W_cp   = W[:, 15:21]
    
    # Fuse Self
    W_self_new, shift_self = fuse_layer_section(W_self, mean_s, std_s)
    
    # Fuse CP
    W_cp_new, shift_cp = fuse_layer_section(W_cp, mean_c, std_c)
    
    # Reassemble
    W_new = np.concatenate([W_self_new, W_cp_new], axis=1)
    b_new = b - shift_self - shift_cp
    
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)

def fuse_normalization_commander(model, rms_stats):
    """
    Fuses RMS stats into CommanderNet.
    Encoder Input: Enemy/Team (13)
    Backbone Input: Self(15) + Latents...
    """
    print("Fusing Commander Normalization...")
    mean_s, std_s = get_ms(rms_stats['self'])
    mean_e, std_e = get_ms(rms_stats['ent']) # Used for both Team and Enemy
    
    # 1. Encoder Layer 0: Linear(13 -> 32)
    layer = model.encoder[0]
    W = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()
    
    W_new, shift = fuse_layer_section(W, mean_e, std_e)
    b_new = b - shift
    
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)
    
    # 2. Backbone Layer 0: Linear(47 -> 64)
    # Input: Self(15) + TM(16) + EN(16)
    layer = model.backbone[0]
    W = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()
    
    W_self = W[:, 0:15]
    W_rest = W[:, 15:] # Latents are normalized by activations, not input stats
    
    W_self_new, shift_self = fuse_layer_section(W_self, mean_s, std_s)
    
    W_new = np.concatenate([W_self_new, W_rest], axis=1)
    b_new = b - shift_self
    
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)

def get_ms(sd):
    mean = sd['mean'].cpu().numpy()
    var = sd['var'].cpu().numpy()
    std = np.sqrt(var + 1e-4)
    return mean, std

def fuse_layer_section(W, mean, std, threshold=0.05):
    W_new = np.zeros_like(W)
    safe_mask = std > threshold
    
    # If std is too small, feature is constant -> W_new=0 (ignore strictly)
    # or we could keep it if we trust the small std, but 0.05 is safe.
    W_new[:, safe_mask] = W[:, safe_mask] / std[None, safe_mask]
    
    shift = np.sum(W_new * mean[None, :], axis=1)
    return W_new, shift

def quantize_weights(actor):
    """
    Extracts and quantizes weights from Universal Actor.
    """
    weights = []
    
    # Pilot (Hidden 64)
    # pilot.net[0], pilot.net[2], pilot.net[4] (NEW), heads
    for layer in [actor.pilot.net[0], actor.pilot.net[2], actor.pilot.net[4], actor.pilot.head_thrust, actor.pilot.head_angle]:
        weights.extend(extract_layer(layer))
        
    # Extra: Role Embedding (No Bias)
    # [2, 16]
    weights.extend(actor.role_embedding.weight.data.cpu().numpy().flatten())
        
    # Commander (Hidden 128)
    # encoder[0], encoder[2], backbone[0], backbone[2]
    # head_shield, head_boost, head_bias_thrust, head_bias_angle
    c_layers = [
        actor.commander.encoder[0], actor.commander.encoder[2],
        actor.commander.backbone[0], actor.commander.backbone[2], actor.commander.backbone[4],
        actor.commander.head_shield, actor.commander.head_boost,
        actor.commander.head_bias_thrust, actor.commander.head_bias_angle
    ]
    for layer in c_layers:
        weights.extend(extract_layer(layer))
        
    # Quantize
    min_val, max_val = min(weights), max(weights)
    scale = max(abs(min_val), abs(max_val)) / 127.0
    
    quantized = []
    for x in weights:
        q = int(round(x / scale))
        q = max(-127, min(127, q))
        quantized.append(q)
        
    return quantized, scale

def extract_layer(layer):
    w = layer.weight.data.cpu().numpy().flatten()
    if layer.bias is not None:
        b = layer.bias.data.cpu().numpy().flatten()
    else:
        b = np.zeros(layer.out_features)
    return np.concatenate([w, b])

def encode_data(quantized_data):
    # Base85 Encoding as per existing logic
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

# Minify the physics code for embedding
def minify_code(code):
    import re
    # Remove comments
    code = re.sub(r'#.*', '', code)
    # Remove empty lines
    code = re.sub(r'\n\s*\n', '\n', code)
    # Remove leading/trailing whitespace
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

SINGLE_FILE_TEMPLATE = """import sys, math
W,H = {WIDTH},{HEIGHT}
SP = 1.0/{WIDTH}.0
SV = 1.0/1000.0

# --- FAST PHYSICS ---
{FAST_PHYSICS}
# --------------------

class N:
    def __init__(self,d,s):
        self.w = self.dec(d,s); self.c = 0
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
    def lin(self,x,i,o,r=False):
        w,b,out=self.gw(i*o),self.gw(o),[]
        for k in range(o):
            a=b[k]
            for j in range(i): a+=x[j]*w[k*i+j]
            out.append(max(0.0,a) if r else a)
        return out
    def emb(self, idx, dim):
        # Embedding: [2, dim] flattened. No bias.
        w = self.gw(2 * dim)
        start = int(idx) * dim
        return w[start : start+dim]

class B(N):
    def f(self,s,t,e,c,role):
        self.c=0
        hp={HP}
        x=s+c; x=self.lin(x,21,hp,True); x=self.lin(x,hp,hp,True); x=self.lin(x,hp,hp,True)
        th_p=self.lin(x,hp,1)[0]; an_p=self.lin(x,hp,1)[0]
        
        # Read Role Emb
        r_emb=self.emb(role, 16)
        
        w1,b1,w2,b2=self.gw(416),self.gw(32),self.gw(512),self.gw(16)
        def enc(inp):
            h=[0.0]*32
            for r in range(32):
                a=b1[r]
                for j in range(13): a+=inp[j]*w1[r*13+j]
                h[r]=max(0.0,a)
            z=[0.0]*16
            for r in range(16):
                a=b2[r]
                for j in range(32): a+=h[j]*w2[r*32+j]
                z[r]=a
            return z
        tm=enc(t); encs=[enc(en) for en in e]
        ctx=[max(x[i] for x in encs) for i in range(16)] if encs else [0.0]*16
        
        hc={HC}
        # Input to Backbone: Self(15) + Team(16) + Enf(16) + Role(16) = 63
        x=s+tm+ctx+r_emb
        x=self.lin(x,63,hc,True); x=self.lin(x,hc,hc,True); x=self.lin(x,hc,hc,True)
        sh=self.lin(x,hc,2); bo=self.lin(x,hc,2)
        b_th=self.lin(x,hc,1)[0]; b_an=self.lin(x,hc,1)[0]
        
        ft = 1.0/(1.0+math.exp(-(th_p + b_th)))
        sum_a = an_p + b_an
        fa = (math.exp(2*sum_a)-1)/(math.exp(2*sum_a)+1)
        
        return [ft, fa, 1 if sh[1]>sh[0] else 0, 1 if bo[1]>bo[0] else 0]

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
                feat=[dpf*SP,dpr*SP,dvf*SV,dvr*SV,math.cos(rr),math.sin(rr),dist,mate,osh,otf*SP,otr*SP,o_run,0.0]
                if o['tm']==p['tm']: otm.extend(feat)
                else: oen.append(feat)
            if not otm: otm=[0.0]*13
            c1=cps[p['n']]; c2=cps[(p['n']+1)%len(cps)]
            cx,cy=c2[0]-c1[0],c2[1]-c1[1]
            cf,cr=tl(cx,cy,p['a'])
            ocp=[ftf,ftr,cf*SP,cr*SP,0.0,0.0]
            
            # Use Single Brain with Role Input
            role_val = 1.0 if run[i] else 0.0
            out = mr.f(oself,otm,oen,ocp, role_val)
            
            rl_th = int(out[0]*100)
            rl_ang = out[1]*18.0
            rl_sh = (out[2]==1)
            rl_bo = (out[3]==1)
            
            # NN output is Relative Angle (Delta).
            # We need Absolute Angle for the Search Candidate (Target Heading).
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
                     # Runner
                     dx, dy = cp_pos[0]-fp['x'], cp_pos[1]-fp['y']
                     t_ang = math.degrees(math.atan2(dy, dx))
                     d_a = abs(t_ang - fp['a'])
                     while d_a > 180: d_a = 360 - d_a
                     score -= d_a * 10
                else:
                     # Blocker: Stick to RL logic more?
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


def export_model(model_path, output_path="submission.py"):
    # Load Model
    # Since we changed defaults to 64, this should be fine
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
        
        fuse_normalization_pilot(agent.actor.pilot, rms_stats)
        fuse_normalization_commander(agent.actor.commander, rms_stats)
        
        # fuse_normalization_pilot(agent.blocker_actor.pilot, rms_stats)
        # fuse_normalization_commander(agent.blocker_actor.commander, rms_stats)
    else:
        print("WARNING: No RMS stats found!")
            
    # Quantize Universal
    q_univ, scale_univ = quantize_weights(agent.actor)
    
    enc_univ = encode_data(q_univ)
    
    # Escape
    univ_esc = enc_univ.replace("\\", "\\\\").replace("\"", "\\\"")
    
    # Retrieve Hidden Dims
    hp = agent.actor.pilot.hidden_dim
    hc = agent.actor.commander.hidden_dim
    
    # Minify Physics
    fast_phys = minify_code(FAST_PHYSICS_CODE)
    
    script = SINGLE_FILE_TEMPLATE.replace("{BLOB_UNIV}", univ_esc)\
        .replace("{SCALE_VAL}", str(scale_univ))\
        .replace("{WIDTH}", str(WIDTH))\
        .replace("{HEIGHT}", str(HEIGHT))\
        .replace("{HP}", str(hp))\
        .replace("{HC}", str(hc))\
        .replace("{FAST_PHYSICS}", fast_phys)
    
    with open(output_path, 'w') as f:
        f.write(script)
        
    print(f"Exported to {output_path}")
    print(f"Total Chars: {len(script)}")
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--out", type=str, default="submission.py")
    args = parser.parse_args()
    
    export_model(args.model, args.out)

if __name__ == "__main__":
    main()
