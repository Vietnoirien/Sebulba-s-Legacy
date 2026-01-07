
import subprocess
import sys

def test_submission():
    print("Testing dummy_submission.py execution...")
    
    # Mock Input:
    # 3 Checkpoints
    # 2 Pods (Self) + 2 Pods (Enemy)
    input_str = """3
1000 1000
5000 5000
10000 10000
3
1 1 0 0 0 1
2 2 0 0 0 1
3 3 0 0 0 1
4 4 0 0 0 1
"""
    # 3 Laps? No, Input format for initialization:
    # Line 1: Laps (int) - Wait, code says: try input() except: C=3 (Checkpoints count)
    # The template 'solve' function:
    # try C=int(input())
    # cps = ...
    # While True: ...
    
    # Let's adjust input string to match exactly
    # 3 (Count)
    # 1000 1000 (CP1)
    # ...
    # Loop:
    # Pod 0: x y vx vy a n
    # Pod 1..3: ...
    
    try:
        process = subprocess.Popen(
            [sys.executable, "dummy_submission.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=input_str, timeout=5)
        
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        
        if "Traceback" in stderr:
            print("FAILED: Traceback found.")
            sys.exit(1)
            
        if not stdout.strip():
            print("FAILED: No output.")
            sys.exit(1)
            
        print("SUCCESS: Generated moves.")
        
    except subprocess.TimeoutExpired:
        print("FAILED: Timeout (Infinite loop?)")
        process.kill()
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_submission()
