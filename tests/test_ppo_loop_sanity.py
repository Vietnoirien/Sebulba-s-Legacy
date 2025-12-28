
import unittest
import sys
import os
sys.path.append(os.getcwd())
import torch
from training.ppo import PPOTrainer
import time
import threading

class TestPPOLoop(unittest.TestCase):
    def test_run_one_iteration(self):
        print("Initializing Trainer...")
        trainer = PPOTrainer(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Override constants for speed
        # We can't easily override global constants in another module without patching,
        # but we can rely on break condition.
        
        stop_event = threading.Event()
        
        def run_target():
            try:
                print("Starting Loop...")
                # We want to run JUST one iteration. 
                # There is no easy argument for "max_iterations".
                # But we can simulate a stop event after a short delay or check `iteration` inside if we could hook it.
                # Since we can't hook, we will let it run and killing it is messy.
                # BETTER: We can mock `TOTAL_TIMESTEPS`? No, it's global.
                
                # Let's just run it and assume if it hits line 273 (DEBUG print) and finishes step loop, it works.
                # Actually, running one full iteration of 128 steps * 4096 envs is heavy for a "unit test".
                # But the user says it "doesn't start".
                
                trainer.train_loop(stop_event=stop_event)
            except Exception as e:
                print(f"Exception: {e}")
                raise e

        t = threading.Thread(target=run_target)
        t.start()
        
        print("Waiting for logical progress...")
        time.sleep(10) # Wait 10s. If it's stuck, it's stuck.
        
        print("Stopping...")
        stop_event.set()
        t.join(timeout=10)
        
        if t.is_alive():
            self.fail("Training loop did not exit cleanly (stuck?)")
        else:
            print("Training loop exited.")

if __name__ == '__main__':
    unittest.main()
