
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import PPOTrainer...")
    from training.ppo import PPOTrainer
    print("Import successful.")
    
    # Optional: Instantiate to check __init__ logic (might be heavy due to CUDA)
    # trainer = PPOTrainer(device='cpu') 
    # print("Instantiation successful.")
    
except Exception as e:
    print(f"Import Failed: {e}")
    import traceback
    traceback.print_exc()
