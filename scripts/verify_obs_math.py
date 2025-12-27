
import torch
import math
import numpy as np

def torch_to_local(vx, vy, angle_deg):
    """
    Rotates vector (vx, vy) by -angle_deg to align with body frame.
    Angle 0 means facing East (1, 0).
    Positive angle is Clockwise? No, standard math is Counter-Clockwise.
    
    Wait, let's check game coordinate system.
    Usually: X is Right, Y is Down (Screen)? Or Cartesian?
    CodingGame usually: 0,0 Top Left.
    Angle 0: East. 
    Angle 90: South? Or North?
    
    Let's assume standard math for now and adjust if Physics engine says otherwise.
    Physics uses: `angle_rad = atan2(dy, dx)`. This is standard Cartesian.
    
    Rotation Matrix for rotating TO Body Frame (which is rotated by Theta):
    We want to rotate the vector by -Theta.
    
    [ cos(-t) -sin(-t) ] [ vx ]
    [ sin(-t)  cos(-t) ] [ vy ]
    
    = [  cos(t) sin(t) ] [ vx ]
    = [ -sin(t) cos(t) ] [ vy ]
    
    New X (Fwd) = vx * cos(t) + vy * sin(t)
    New Y (Right? Left?) = -vx * sin(t) + vy * cos(t)
    
    If t=0 (Face East), vx=1, vy=0 -> Fwd=1, Side=0. Correct.
    If t=90 (Face South?), vx=0, vy=1 -> Fwd=1.
       Cos(90)=0, Sin(90)=1.
       Fwd = 0*0 + 1*1 = 1. Correct.
    """
    rad = torch.deg2rad(angle_deg)
    c = torch.cos(rad)
    s = torch.sin(rad)
    
    # Batch supported
    v_fwd = vx * c + vy * s
    v_right = -vx * s + vy * c
    
    return v_fwd, v_right

def python_to_local(vx, vy, angle_deg):
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    
    v_fwd = vx * c + vy * s
    v_right = -vx * s + vy * c
    
    return v_fwd, v_right

def run_verify():
    print("Verifying Observation Math Consistency...")
    
    # Test Cases
    cases = [
        (100.0, 0.0, 0.0),    # Moving East, Facing East -> Fwd=100
        (0.0, 100.0, 90.0),   # Moving South, Facing South -> Fwd=100
        (100.0, 0.0, 90.0),   # Moving East, Facing South -> Right=-100 (West is Left of South... wait. East is Left of South?)
                              # If Facing South (Down), East is Left. 
                              # Formula: Right = -100 * 1 + 0 = -100. 
                              # So "Right" means "Starboard"? Or "Port"?
                              # Let's check.
                              # Facing South. Right is West?
                              # Map: N(Up), E(Right), S(Down), W(Left).
                              # Facing S. Right of me is W.
                              # Moving E. E is Left of me.
                              # So should be "Negative Right".
                              # Result -100. Correct.
    ]
    
    for vx, vy, ang in cases:
        # Torch
        tvx = torch.tensor([vx])
        tvy = torch.tensor([vy])
        tang = torch.tensor([ang])
        
        tf, tr = torch_to_local(tvx, tvy, tang)
        
        # Python
        pf, pr = python_to_local(vx, vy, ang)
        
        print(f"Case ({vx}, {vy}) @ {ang}deg:")
        print(f"  Torch:  Fwd={tf.item():.4f}, Right={tr.item():.4f}")
        print(f"  Python: Fwd={pf:.4f}, Right={pr:.4f}")
        
        assert abs(tf.item() - pf) < 1e-4
        assert abs(tr.item() - pr) < 1e-4
        
    print("SUCCESS: Math is consistent.")

if __name__ == "__main__":
    run_verify()
