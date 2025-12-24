
import time
import torch
import numpy as np
import threading
from training.ppo import PPOTrainer

def verify_performance():
    print("Initializing PPOTrainer for Performance Check...")
    # Use 'cpu' or 'cuda' depending on system. Assuming cuda if available for realistic test, but code handles both.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = PPOTrainer(device=device)
    
    # Telemetry Callback Mock
    telemetry_counts = 0
    
    def mock_telemetry_callback(*args, **kwargs):
        nonlocal telemetry_counts
        telemetry_counts += 1
        
    print("Starting Training Loop (Short Run)...")
    stop_event = threading.Event()
    
    # Run in a separate thread to allow us to stop it
    t = threading.Thread(target=trainer.train_loop, kwargs={'stop_event': stop_event, 'telemetry_callback': mock_telemetry_callback})
    t.start()
    
    # Let it run for 10 seconds
    try:
        start_time = time.time()
        duration = 10.0
        time.sleep(duration)
        stop_event.set()
        t.join()
        end_time = time.time()
        
        real_duration = end_time - start_time
        
        # Calculate Stats
        total_steps = trainer.iteration * 512 * 4096 # Total atomic steps? No.
        # PPOTrainer doesn't track global SPS simply in a variable we can access from outside easily 
        # except maybe by reading the logs it prints? 
        # But we can check `telemetry_counts`.
        
        print(f"Test Duration: {real_duration:.2f}s")
        print(f"Telemetry Callbacks: {telemetry_counts}")
        print(f"Telemetry Frequency: {telemetry_counts / real_duration:.2f} Hz")
        
        # Expected Frequency: ~20Hz
        if 15.0 < (telemetry_counts / real_duration) < 25.0:
            print("PASS: Telemetry Throttling Frequency is within target (~20Hz).")
        else:
            print(f"FAIL: Telemetry Frequency {telemetry_counts / real_duration:.2f} Hz is out of expected range (20Hz).")
            
    except Exception as e:
        print(f"Error during test: {e}")
        stop_event.set()
        t.join()

if __name__ == "__main__":
    verify_performance()
