import numpy as np
import struct
import base64
from training.telemetry import TelemetryWorker

def test_packing():
    print("Testing Telemetry Packing...")
    
    # Mock Data matching session.py output
    data = {
        "step": 100,
        "env_idx": 0,
        "pos": np.zeros((4, 2), dtype=np.float32),
        "vel": np.zeros((4, 2), dtype=np.float32),
        "angle": np.zeros(4, dtype=np.float32),
        "laps": np.zeros(4, dtype=np.int32),
        "next_cps": np.zeros(4, dtype=np.int32),
        "rewards": np.zeros(4, dtype=np.float32),
        "actions": np.zeros((4, 4), dtype=np.float32),
        "dones": False,
        "collision_flags": np.zeros(4, dtype=np.float32)
    }
    
    worker = TelemetryWorker(None, None, None)
    
    try:
        frame = worker.pack_frame(data)
        print(f"Pack Success! Length: {len(frame)}")
        assert len(frame) == 192, f"Expected 192 bytes, got {len(frame)}"
        print("Size Check Passed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pack Failed: {e}")

if __name__ == "__main__":
    test_packing()
