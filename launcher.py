import subprocess
import signal
import sys
import time
import os
import webbrowser
from pathlib import Path

def cleanup(backend_proc, frontend_proc):
    print("\nStopping services...")
    
    # Terminate frontend
    if frontend_proc and frontend_proc.poll() is None:
        print("Stopping Frontend...")
        try:
            os.killpg(os.getpgid(frontend_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    
    # Terminate backend
    if backend_proc and backend_proc.poll() is None:
        print("Stopping Backend...")
        try:
            os.killpg(os.getpgid(backend_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    # Wait for graceful exit with Timeout
    print("Waiting for grace period (5s)...")
    
    if frontend_proc:
        try:
            frontend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Frontend unresponsive. Force killing...")
            try:
                os.killpg(os.getpgid(frontend_proc.pid), signal.SIGKILL)
            except: pass

    if backend_proc:
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Backend unresponsive. Force killing...")
            try:
                os.killpg(os.getpgid(backend_proc.pid), signal.SIGKILL)
            except: pass
        
    print("Services stopped.")
    sys.exit(0)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Sebulba's Legacy Launcher")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling logs")
    args = parser.parse_args()

    print("Starting Sebulba's Legacy System...")
    
    project_root = Path(__file__).parent.absolute()
    
    # Launch Backend
    print("Launching Backend...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    if args.profile:
        print("-> Profiling Enabled")
        env["ENABLE_PROFILING"] = "1"
    
    backend_cmd = [
        ".venv/bin/python",
        "web/backend/main.py"
    ]
    
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=project_root,
        env=env,
        preexec_fn=os.setsid # Create new process group
    )
    
    # Launch Frontend
    print("Launching Frontend...")
    frontend_dir = project_root / "web/frontend"
    
    frontend_cmd = [
        "npm",
        "run",
        "dev"
    ]
    
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=frontend_dir,
        preexec_fn=os.setsid # Create new process group
    )
    
    def signal_handler(sig, frame):
        cleanup(backend_proc, frontend_proc)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("System running. Press Ctrl+C to stop.")
    
    # Open Browser
    import webbrowser
    print("Opening browser...")
    # Give a moment for services to bind
    time.sleep(1.5) 
    webbrowser.open("http://localhost:5173")
    
    try:
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if backend_proc.poll() is not None:
                print("Backend process terminated unexpectedly.")
                cleanup(backend_proc, frontend_proc)
            if frontend_proc.poll() is not None:
                print("Frontend process terminated unexpectedly.")
                cleanup(backend_proc, frontend_proc)
                
    except KeyboardInterrupt:
        cleanup(backend_proc, frontend_proc)

if __name__ == "__main__":
    main()
