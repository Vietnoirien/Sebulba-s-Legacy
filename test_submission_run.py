import subprocess
import time
import sys

def test_submission():
    print("Testing submission.py...")
    
    # Init inputs
    # Laps(3)
    # CheckpointCount(3)
    # CP1, CP2, CP3
    init_input = "3\n3\n0 0\n1000 0\n1000 1000\n"
    
    # 1 Turn Input
    # 2 My Pods
    # 2 Opponent Pods
    # x y vx vy a n
    turn_input = ""
    for _ in range(2): turn_input += "100 100 0 0 0 1\n"
    for _ in range(2): turn_input += "900 900 0 0 0 1\n"
    
    full_input = init_input + turn_input
    
    start_time = time.time()
    
    process = subprocess.Popen(
        ['python3', 'submission.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        stdout, stderr = process.communicate(input=full_input, timeout=5)
        
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        
        if process.returncode != 0:
            print(f"FAILED: Return Code {process.returncode}")
            return
            
        lines = stdout.strip().split('\n')
        if len(lines) >= 2:
            print("SUCCESS: Received output lines.")
            print(f"Time taken: {time.time() - start_time:.4f}s")
        else:
            print(f"FAILED: Expected at least 2 lines, got {len(lines)}")
            
    except subprocess.TimeoutExpired:
        process.kill()
        print("FAILED: Timeout Expired (5s)")
        out, err = process.communicate()
        print("STDERR after timeout:", err)

if __name__ == "__main__":
    test_submission()
