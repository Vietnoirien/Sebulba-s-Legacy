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
    print("Fusing Pilot Normalization...")
    mean_s, std_s = get_ms(rms_stats['self'])
    mean_c, std_c = get_ms(rms_stats['cp'])
    
    layer = pilot.pilot_embed[0]
    W = layer.weight.data.cpu().numpy()
    b = layer.bias.data.cpu().numpy()
    
    W_self = W[:, 0:15]
    W_cp   = W[:, 15:25]
    
    W_self_new, shift_self = fuse_layer_section(W_self, mean_s, std_s)
    W_cp_new, shift_cp = fuse_layer_section(W_cp, mean_c, std_c)
    
    W_new = np.concatenate([W_self_new, W_cp_new], axis=1)
    b_new = b - shift_self - shift_cp
    
    layer.weight.data = torch.tensor(W_new, dtype=torch.float32)
    layer.bias.data = torch.tensor(b_new, dtype=torch.float32)

def fuse_normalization_commander(actor, rms_stats):
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
    
    # 2. Backbone [0] (Input Dim varies, first 15 is self)
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
    weights = []
    
    # 1. Pilot Embed
    weights.extend(extract_layer(actor.pilot_embed[0]))
    weights.extend(extract_layer(actor.pilot_embed[2]))
    
    # 2. Enemy Enc
    weights.extend(extract_layer(actor.enemy_encoder[0]))
    weights.extend(extract_layer(actor.enemy_encoder[2]))
    
    # 3. Role Emb
    weights.extend(actor.role_embedding.weight.data.cpu().numpy().flatten())
    
    # 4. Map Encoder
    weights.extend(extract_layer(actor.map_encoder.input_proj))
    
    enc = actor.map_encoder.transformer.layers[0]
    # MHA In Proj
    weights.extend(enc.self_attn.in_proj_weight.data.cpu().numpy().flatten())
    weights.extend(enc.self_attn.in_proj_bias.data.cpu().numpy().flatten())
    # MHA Out Proj
    weights.extend(enc.self_attn.out_proj.weight.data.cpu().numpy().flatten())
    weights.extend(enc.self_attn.out_proj.bias.data.cpu().numpy().flatten())
    # Norm 1
    weights.extend(extract_layer(enc.norm1))
    # FF Lin 1
    weights.extend(extract_layer(enc.linear1))
    # FF Lin 2
    weights.extend(extract_layer(enc.linear2))
    # Norm 2
    weights.extend(extract_layer(enc.norm2))

    # 5. Commander Backbone
    weights.extend(extract_layer(actor.commander_backbone[0]))
    weights.extend(extract_layer(actor.commander_backbone[2]))
    
    # 6. LSTM
    lstm = actor.lstm
    if hasattr(lstm, 'weight_ih_l0'):
        wi = lstm.weight_ih_l0.data.cpu().numpy().flatten()
        wh = lstm.weight_hh_l0.data.cpu().numpy().flatten()
        bi = lstm.bias_ih_l0.data.cpu().numpy()
        bh = lstm.bias_hh_l0.data.cpu().numpy()
    else:
        wi = lstm.ih.weight.data.cpu().numpy().flatten()
        wh = lstm.hh.weight.data.cpu().numpy().flatten()
        bi = lstm.ih.bias.data.cpu().numpy()
        bh = lstm.hh.bias.data.cpu().numpy()
        
    b = bi + bh
    weights.extend(wi)
    weights.extend(wh)
    weights.extend(b)
    
    # 7. Heads
    weights.extend(extract_layer(actor.head_thrust))
    weights.extend(extract_layer(actor.head_angle))
    weights.extend(extract_layer(actor.head_shield))
    weights.extend(extract_layer(actor.head_boost))
    
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
def clip(v,l,h): return max(l,min(h,v))
def sim(pods,cps,acts):
    for i,p in enumerate(pods):
        da=clip(acts[i][1],-18.0,18.0); p['a']+=da
        if p['a']>180: p['a']-=360
        elif p['a']<=-180: p['a']+=360
        th=clip(acts[i][0],0,100)
        if acts[i][2]: p['s']=4; p['m']=10.0; th=0
        else:
             if p['s']>0: p['s']-=1; th=0
             p['m']=10.0 if p['s']==3 else 1.0
        r=math.radians(p['a'])
        p['vx']+=th*math.cos(r); p['vy']+=th*math.sin(r)
        p['x']+=p['vx']; p['y']+=p['vy']
    for _ in range(2): 
        prs=[]
        for i in range(4):
            for j in range(i+1,4):
                 dx=pods[j]['x']-pods[i]['x']; dy=pods[j]['y']-pods[i]['y']; d2=dx*dx+dy*dy
                 if d2<640000:
                     d=math.sqrt(d2); d=1e-4 if d<1e-4 else d
                     nx,ny=dx/d,dy/d; p=(pods[i]['vx']-pods[j]['vx'])*nx+(pods[i]['vy']-pods[j]['vy'])*ny
                     m1,m2=pods[i].get('m',1.0),pods[j].get('m',1.0); f=p/(1/m1+1/m2)
                     if f<120.0 and p>0: f=120.0
                     elif p<=0: f=0
                     jx,jy=-f*nx,-f*ny; ov=800-d; sx,sy=nx*ov*0.5,ny*ov*0.5
                     prs.append((i,j,jx,jy,sx,sy,m1,m2))
        for i,j,jx,jy,sx,sy,m1,m2 in prs:
             pods[i]['vx']+=jx/m1; pods[i]['vy']+=jy/m1; pods[i]['x']-=sx; pods[i]['y']-=sy
             pods[j]['vx']-=jx/m2; pods[j]['vy']-=jy/m2; pods[j]['x']+=sx; pods[j]['y']+=sy
    for p in pods:
        p['vx']*=0.85; p['vy']*=0.85; p['x']=round(p['x']); p['y']=round(p['y'])
        p['vx']=int(p['vx']); p['vy']=int(p['vy'])
        cp=cps[p['n']]
        if (p['x']-cp[0])**2+(p['y']-cp[1])**2<360000:
            p['n']=(p['n']+1)
            if p['n']>=len(cps): p['n']=0; p['l']+=1
            p['to']=100
        else: p['to']-=1
    return pods
"""

SINGLE_FILE_TEMPLATE = """import sys,math
SP=1.0/{WIDTH}.0; SV=1.0/1000.0
{FAST_PHYSICS}
class N:
    def __init__(self,d,s): 
        self.w=self.dc(d,s); self.c=0; self.h=[0.0]*{LSTM}; self.C=[0.0]*{LSTM}
    def dc(self,b,s):
        w,v,cnt=[],0,0
        for c in b:
            v=v*85+(ord(c)-33); cnt+=1
            if cnt==5:
                for i in range(4): x=(v>>(24-i*8))&0xFF; w.append((x-256 if x>127 else x)*s)
                v,cnt=0,0
        return w
    def gw(self,n): 
        # if self.c+n > len(self.w): print(f"DEBUG: OOM! gw({{n}}) at {{self.c}}, len {{len(self.w)}}"); return [0.0]*n
        r=self.w[self.c:self.c+n]; self.c+=n; return r
    def l(self,x,i,o,b=True,r=False):
        w,bs,out=self.gw(i*o),self.gw(o) if b else [0.0]*o,[]
        for k in range(o):
            a=bs[k]
            for j in range(i): a+=x[j]*w[k*i+j]
            out.append(max(0.0,a) if r else a)
        return out
    def ls(self,x):
        wi=self.l(x,96,{GATES},b=False); wh=self.l(self.h,{LSTM},{GATES},b=False); b=self.gw({GATES})
        nc,nh=[],[]
        for k in range({LSTM}):
            i,f,g,o=k,{LSTM}+k,2*{LSTM}+k,3*{LSTM}+k
            gi=wi[i]+wh[i]+b[i]; gf=wi[f]+wh[f]+b[f]; gg=wi[g]+wh[g]+b[g]; go=wi[o]+wh[o]+b[o]
            si=1.0/(1.0+math.exp(-gi)); sf=1.0/(1.0+math.exp(-gf)); so=1.0/(1.0+math.exp(-go)); tg=math.tanh(gg)
            c=sf*self.C[k]+si*tg; h=so*math.tanh(c); nc.append(c); nh.append(h)
        self.C,self.h=nc,nh; return nh
    def mr(self,m):
        gw=self.gw; we,be=gw(64),gw(32); wi,bi=gw(3072),gw(96); wo,bo=gw(1024),gw(32)
        gn1,bn1=gw(32),gw(32); wl1,bl1=gw(2048),gw(64); wl2,bl2=gw(2048),gw(32); gn2,bn2=gw(32),gw(32)
        def L(x,w,b,i,o):
            r=[]
            for k in range(o):
                a=b[k]
                for j in range(i): a+=x[j]*w[k*i+j]
                r.append(a)
            return r
        def Nm(x,g,b):
            m=sum(x)/len(x); v=sum((z-m)**2 for z in x)/len(x); s=math.sqrt(v+1e-5)
            return [(x[i]-m)/s*g[i]+b[i] for i in range(len(x))]
        if len(m)>6: m=m[:6]
        sq=[L(c,we,be,2,32) for c in m]; sk=[s[:] for s in sq]; qk=[L(s,wi,bi,32,96) for s in sq]
        at=[]; sc=0.25
        for t in range(len(sq)):
            row=[0.0]*32
            for h in range(2):
                qs,ke,ve=h*16,32+h*16,64+h*16; q=qk[t][qs:qs+16]; scs=[]
                for z in range(len(sq)): k=qk[z][ke:ke+16]; scs.append(sum(q[j]*k[j] for j in range(16))*sc)
                mx=max(scs); exs=[math.exp(s-mx) for s in scs]; sm=sum(exs); pr=[e/sm for e in exs]
                hv=[0.0]*16
                for z,p in enumerate(pr):
                     v=qk[z][ve:ve+16]
                     for j in range(16): hv[j]+=p*v[j]
                for j in range(16): row[h*16+j]=hv[j]
            at.append(row)
        pj=[L(a,wo,bo,32,32) for a in at]; n1=[Nm([sk[i][j]+pj[i][j] for j in range(32)],gn1,bn1) for i in range(len(sq))]
        sk2=[x[:] for x in n1]; ff=[]
        for x in n1: l1=L(x,wl1,bl1,32,64); l1=[max(0.0,v) for v in l1]; ff.append(L(l1,wl2,bl2,64,32))
        n2=[Nm([sk2[i][j]+ff[i][j] for j in range(32)],gn2,bn2) for i in range(len(sq))]
        return [max(r[j] for r in n2) for j in range(32)] if n2 else [0.0]*32
    def f(self,s,tm,en,cp,mo,rv,d):
        self.c=0
        pe=self.l(s+cp,25,64,r=True); pe=self.l(pe,64,48,r=True)
        ew1,eb1=self.gw(448),self.gw(32); ew2,eb2=self.gw(512),self.gw(16)
        def ec(x):
            h=[]
            for k in range(32):
                a=eb1[k]
                for j in range(14): a+=x[j]*ew1[k*14+j]
                h.append(max(0.0,a))
            o=[]
            for k in range(16):
                a=eb2[k]
                for j in range(32): a+=h[j]*ew2[k*32+j]
                o.append(a)
            return o
        t_lat=ec(tm)
        e_ctx=[-9e9]*16
        if not en: e_ctx=[0.0]*16
        else:
            for ef in en:
                o=ec(ef)
                for k in range(16): 
                    if o[k]>e_ctx[k]: e_ctx[k]=o[k]
        re_w=self.gw(32); re=re_w[int(rv)*16:(int(rv)+1)*16]
        me=self.mr(mo)
        x=s+t_lat+e_ctx+re+me
        cb=self.l(x,95,96,r=True); cb=self.l(cb,96,48,r=True); o=self.ls(pe+cb)
        th=self.l(o,{LSTM},1)
        ang=self.l(o,{LSTM},1)
        sh=self.l(o,{LSTM},2)
        bo=self.l(o,{LSTM},2)
        r_th=1.0/(1.0+math.exp(-th[0])); r_ang=math.tanh(ang[0])
        return r_th, r_ang, 1 if sh[1]>sh[0] else 0, 1 if bo[1]>bo[0] else 0
blob="{BLOB_UNIV}"; scale={SCALE_VAL}
def tl(vx,vy,a):
    r=math.radians(a); c,s=math.cos(r),math.sin(r)
    return vx*c+vy*s, -vx*s+vy*c
def solve():
    b=N(blob,scale)
    try: 
        input() # laps
        C=int(input()) # checkpoint_count
    except: C=3
    cps=[list(map(int,input().split())) for _ in range(C)]
    lps,pnc,to,scd,avl=[0]*4,[1]*4,[100]*4,[0]*4,[True,True]
    while True:
        pds=[]
        try:
            l=input()
            if not l: break
            pts=list(map(int,l.split())); pds.append({'x':pts[0],'y':pts[1],'vx':pts[2],'vy':pts[3],'a':pts[4],'n':pts[5],'id':0,'tm':0,'s':scd[0],'m':10.0 if scd[0]==3 else 1.0,'to':to[0],'l':lps[0]})
        except EOFError: break
        for i in range(1,4):
             pts=list(map(int,input().split()))
             tm=0 if i<2 else 1; s=scd[i] if i<2 else 0
             pds.append({'x':pts[0],'y':pts[1],'vx':pts[2],'vy':pts[3],'a':pts[4],'n':pts[5],'id':i,'tm':tm,'s':s,'m':1.0,'to':to[i],'l':lps[i]})
        for i in range(4):
            p=pds[i]
            if p['n']!=pnc[i]:
                if p['n']==1 and pnc[i]==0: lps[i]+=1
                to[i]=100
            else: to[i]-=1
            pnc[i]=p['n']; pds[i]['l']=lps[i]; pds[i]['to']=to[i]
        scrs=[]
        for i in range(4):
            p=pds[i]; cp=cps[p['n']] if p['n']<len(cps) else cps[0]
            d=math.sqrt((p['x']-cp[0])**2+(p['y']-cp[1])**2); ec=C if p['n']==0 else p['n']
            scrs.append(lps[i]*50000+ec*500+(20000-d)/20)
        run=[False]*4
        if scrs[0]>=scrs[1]: run[0]=True
        else: run[1]=True
        if scrs[2]>=scrs[3]: run[2]=True
        else: run[3]=True
        for i in range(2):
            p=pds[i]; vf,vr=tl(p['vx'],p['vy'],p['a']); fvf,fvr=vf*SV,vr*SV
            tar=cps[p['n']]; gx,gy=tar[0]-p['x'],tar[1]-p['y']; tf,tr=tl(gx,gy,p['a'])
            ftf,ftr=tf*SP,tr*SP; ftd=math.sqrt(gx**2+gy**2)*SP; ds=ftd+1e-6
            fac,fas=(ftf/SP)/(ds/SP),(ftr/SP)/(ds/SP)
            fsh=scd[i]/3.0; fbo=1.0 if avl[0] else 0.0
            fto=to[i]/100.0; fla=lps[i]/3.0; fle=1.0 if run[i] else 0.0
            vm=math.sqrt(p['vx']**2+p['vy']**2)*SV
            rnk=0
            for s in scrs: 
                if s>scrs[i]: rnk+=1
            flr=rnk/3.0
            os=[fvf,fvr,ftf,ftr,ftd,fac,fas,fsh,fbo,fto,fla,fle,vm,0.0,flr]
            ot,oe=[],[]
            for j in range(4):
                if i==j: continue
                o=pds[j]; dx,dy=o['x']-p['x'],o['y']-p['y']; dvx,dvy=o['vx']-p['vx'],o['vy']-p['vy']
                dpf,dpr=tl(dx,dy,p['a']); dvf,dvr=tl(dvx,dvy,p['a'])
                ra=o['a']-p['a']; rr=math.radians(ra); dst=math.sqrt(dx**2+dy**2)*SP
                mt=1.0 if o['tm']==p['tm'] else 0.0
                otc=cps[o['n']]; otx,oty=otc[0]-p['x'],otc[1]-p['y']; otf,otr=tl(otx,oty,p['a'])
                orn=1.0 if run[j] else 0.0
                ornk=0
                for s in scrs:
                    if s>scrs[j]: ornk+=1
                ft=[dpf*SP,dpr*SP,dvf*SV,dvr*SV,math.cos(rr),math.sin(rr),dst,mt,0.0,otf*SP,otr*SP,orn,ornk/3.0,o['to']/100.0]
                if o['tm']==p['tm']: ot.extend(ft)
                else: oe.append(ft)
            if not ot: ot=[0.0]*14
            om=[]
            cn=len(cps); sn=p['n']
            for k in range(cn):
                 idx=(sn+k)%cn; cx,cy=cps[idx][0]-p['x'],cps[idx][1]-p['y']
                 cf,cr=tl(cx,cy,p['a']); om.append([cf*SP,cr*SP])
            rv=1.0 if run[i] else 0.0
            # CP Obs (10 dim)
            cn=len(cps); sn=p['n']
            cp1=cps[sn]; cp2=cps[(sn+1)%cn]; cp3=cps[(sn+2)%cn]
            # v12
            g12x,g12y=cp2[0]-cp1[0],cp2[1]-cp1[1]; v12f,v12r=tl(g12x,g12y,p['a'])
            # v23
            g23x,g23y=cp3[0]-cp2[0],cp3[1]-cp2[1]; v23f,v23r=tl(g23x,g23y,p['a'])
            # Heuristics
            # Corner Cos
            # t_vec is (gx, gy) from line 374. Need normalized.
            # t_vec local is (tf, tr).
            ga=math.sqrt(gx**2+gy**2)+1e-5; g12d=math.sqrt(g12x**2+g12y**2)+1e-5
            d01x,d01y=gx/ga,gy/ga; d12x,d12y=g12x/g12d,g12y/g12d
            # Dot product independent of frame, use global
            ccos=d01x*d12x+d01y*d12y
            msp=(ccos+1.0)*0.5
            # Left
            done=lps[i]*cn+sn; tot=3*cn; left=tot-done
            lft=left/20.0
             
            ocp=[tf*SP,tr*SP, v12f*SP,v12r*SP, v23f*SP,v23r*SP, lft, ccos, msp, 0.0] 
            r_th,r_ang,r_sh,r_bo=b.f(os,ot,oe,ocp,om,rv,False)
            rl_th=int(r_th*100); rl_ang=r_ang*18.0; rl_sh=(r_sh==1)
            c_ang=pds[i]['a']; r_abs=c_ang+rl_ang; bst=(rl_th,r_abs,rl_sh,False)
            cands=[(rl_th,r_abs,rl_sh)]
            if not rl_sh:
                for da in [0,3,-3,6,-6,12,-12]: cands.append((100,r_abs+da,False)); cands.append((0,r_abs,False))
            bsc=-9e9
            for th,ang,sh in cands:
                sp=[x.copy() for x in pds]; acts=[[0,0,0]]*4; acts[i]=[th,ang-sp[i]['a'],sh]
                fp=sim(sp,cps,acts)[i]
                sc=0; cpi=fp['n']; cpp=cps[cpi]; dst=math.sqrt((fp['x']-cpp[0])**2+(fp['y']-cpp[1])**2)
                sc+=(fp['n']+fp['l']*len(cps))*50000-dst
                if run[i]:
                     dx,dy=cpp[0]-fp['x'],cpp[1]-fp['y']; ta=math.degrees(math.atan2(dy,dx))
                     da=abs(ta-fp['a']); 
                     while da>180: da=360-da
                     sc-=da*10
                else: da=abs(ang-rl_ang); sc-=da*50
                if fp['x']<400 or fp['x']>15600 or fp['y']<400 or fp['y']>8600: sc-=100000
                if sc>bsc: bsc=sc; bst=(th,ang,sh,(r_bo==1))
            fth,fang,fsh,fbo=bst
            tx=int(p['x']+1e4*math.cos(math.radians(fang))); ty=int(p['y']+1e4*math.sin(math.radians(fang)))
            pw=str(int(fth))
            if fsh and scd[i]==0: pw="SHIELD"; scd[i]=4
            elif fbo and avl[0] and not fsh: pw="BOOST"; avl[0]=False
            if scd[i]>0: scd[i]-=1
            print(f"{tx} {ty} {pw}")
    if scd[0]>0: scd[0]-=1
    if scd[1]>0: scd[1]-=1
    for i in range(2): 
        if avl[i]: avl[i]=False
solve()
"""

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
