
import torch
import threading
import time
from training.ppo import PPOTrainer
from config import STAGE_SOLO

def test_masking():
    print("Initializing Trainer...")
    trainer = PPOTrainer(device='cpu') # Use CPU for test to match any env
    trainer.env.curriculum_stage = STAGE_SOLO
    
    print(f"Stage: {trainer.env.curriculum_stage}")
    
    stop_event = threading.Event()
    
    def run_trainer():
        try:
            # Monkey patch Total Timesteps to be small?
            # Or just let it run one iter.
            trainer.train_loop(stop_event=stop_event)
        except Exception as e:
            print(f"Trainer Error: {e}")
            import traceback
            traceback.print_exc()

    t = threading.Thread(target=run_trainer)
    t.start()
    
    print("Trainer running...")
    time.sleep(10) # Let it run for a bit
    print("Stopping...")
    stop_event.set()
    t.join(timeout=5)
    
    if t.is_alive():
        print("Trainer stuck!")
    else:
        print("Trainer finished gracefully.")

if __name__ == "__main__":
    test_masking()
