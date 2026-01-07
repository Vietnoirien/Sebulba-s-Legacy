
import torch
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.getcwd())

from models.deepsets import PodAgent
from export import export_model

def make_dummy():
    print("Creating Dummy LSTM Agent...")
    # Initialize random agent
    agent = PodAgent(hidden_dim=128, lstm_hidden=48)
    agent.eval()
    
    # Create temp directory
    os.makedirs("data/dummy", exist_ok=True)
    model_path = "data/dummy/agent_dummy.pt"
    
    # Save model
    print(f"Saving dummy model to {model_path}...")
    torch.save(agent.state_dict(), model_path)
    
    # Export
    output_path = "dummy_submission.py"
    print(f"Exporting to {output_path}...")
    
    # We use the imported export_model function
    # It handles loading and export logic
    export_model(model_path, output_path)
    
    print("Done!")

if __name__ == "__main__":
    make_dummy()
