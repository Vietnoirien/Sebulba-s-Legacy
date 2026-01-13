
import torch
import os
import sys

def check_dims(path, rms_path):
    print(f"Checking {path}")
    if os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        
        if 'state_dict' in state:
             sd = state['state_dict']
        else:
             sd = state
             
        keys = list(sd.keys())
        print(f"Keys found: {len(keys)}")
        
        target_key = 'actor.pilot_embed_runner.0.weight'
        if target_key not in sd:
             # Try without actor prefix?
             target_key = 'pilot_embed_runner.0.weight'
             
        if target_key in sd:
             w = sd[target_key]
             print(f"Pilot Input Dim (from {target_key}): {w.shape[1]}") # Expect 25 (15+10)
             
             # RMS
             if os.path.exists(rms_path):
                 rms = torch.load(rms_path, map_location='cpu')
                 if 'self' in rms:
                     mean_s = rms['self']['mean']
                     print(f"RMS Self Dim: {mean_s.shape[0]}")
                     
                     # Logic check
                     if w.shape[1] == 25:
                          # Self is 15.
                          w_self = 15
                          rms_self = mean_s.shape[0]
                          print(f"Model Self: {w_self} vs RMS Self: {rms_self}")
                          if w_self != rms_self:
                              print("MISMATCH DETECTED!")
                          else:
                              print("MATCH.")
                 else:
                     print("RMS 'self' key missing.")
             else:
                 print(f"RMS File not found at {rms_path}")
        else:
             print(f"Key {target_key} not found. Sample keys: {keys[:5]}")
                 
    else:
        print("Model Path not found.")

if __name__ == "__main__":
    # Find latest gen
    import glob
    gens = glob.glob("data/stage_4/gen_*")
    if not gens:
         print("No gens found.")
    else:
         latest = max(gens, key=os.path.getmtime)
         agent = os.path.join(latest, "agent_0.pt")
         rms = os.path.join(latest, "rms_stats.pt")
         check_dims(agent, rms)
