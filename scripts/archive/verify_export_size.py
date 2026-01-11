import torch
import os
import subprocess
from models.deepsets import PodAgent
from export import export_model

def verify_size():
    print("Initializing Dummy Agent (Dim 15)...")
    agent = PodAgent(hidden_dim=160)
    
    # Save dummy
    dummy_path = "dummy_model.pt"
    torch.save(agent.state_dict(), dummy_path)
    
    print("Exporting...")
    out_path = "submission_test.py"
    try:
        export_model(dummy_path, out_path)
        
        # Check size (Characters)
        result = subprocess.run(['wc', '-m', out_path], capture_output=True, text=True)
        print(f"WC Output: {result.stdout.strip()}")
        
        try:
            chars = int(result.stdout.strip().split()[0])
            limit = 100000
            if chars < limit:
                print(f"SUCCESS: {chars} chars < {limit}")
            else:
                print(f"FAILURE: {chars} chars >= {limit}")
        except:
             print("Could not parse wc output.")
            
    except Exception as e:
        print(f"Export failed: {e}")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
        # Keep submission for inspection

if __name__ == "__main__":
    verify_size()
