import torch
import os
import sys
from models.deepsets import PodAgent
from export import export_model

def test_architecture():
    print("Initializing PodAgent...")
    agent = PodAgent() # Defaults: Pilot 32, Commander 64
    
    # Mock Inputs
    B = 2
    self_obs = torch.randn(B, 15)
    tm_obs = torch.randn(B, 13)
    en_obs = torch.randn(B, 2, 13) # N=2 enemies
    cp_obs = torch.randn(B, 6)
    
    print("Running Forward Pass...")
    (th, an), std, (sh, bo) = agent.runner_actor(self_obs, tm_obs, en_obs, cp_obs)
    
    print(f"Output Shapes: Thrust {th.shape}, Angle {an.shape}, Shield {sh.shape}")
    assert th.shape == (B, 1)
    
    print("Forward Pass Successful.")
    
    # Save dummy model
    model_path = "test_model.pt"
    torch.save(agent.state_dict(), model_path)
    
    # Create dummy RMS stats
    rms = {
        'self': {'mean': torch.zeros(15), 'var': torch.ones(15)},
        'cp': {'mean': torch.zeros(6), 'var': torch.ones(6)},
        'ent': {'mean': torch.zeros(13), 'var': torch.ones(13)}
    }
    torch.save(rms, "rms_stats.pt")
    
    print("Exporting...")
    out_path = export_model(model_path, "submission_test.py")
    
    # Check size
    size = os.path.getsize(out_path)
    print(f"Submission Size: {size} bytes")
    
    if size > 100000:
        print("FAIL: Size exceeds 100k limit!")
        sys.exit(1)
    else:
        print("PASS: Size within limit.")
        
    # Check syntax
    import py_compile
    py_compile.compile(out_path, doraise=True)
    print("PASS: Valid Python syntax.")
    
    # Cleanup
    # os.remove(model_path)
    # os.remove("rms_stats.pt")
    # os.remove(out_path)

if __name__ == "__main__":
    test_architecture()
