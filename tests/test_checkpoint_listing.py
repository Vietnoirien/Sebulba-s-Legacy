import os
import shutil
import json
import asyncio
from unittest.mock import MagicMock
from web.backend.main import list_generations
from training.session import TrainingSession

async def test_listing_and_loading():
    print("--- Testing Checkpoint Listing & Loading ---")
    
    # 1. Setup Mock Checkpoints
    base_data = "data"
    stage_dir = os.path.join(base_data, "stage_99")
    gen_dir = os.path.join(stage_dir, "gen_123")
    
    os.makedirs(gen_dir, exist_ok=True)
    
    # Create dummy agent
    with open(os.path.join(gen_dir, "agent_0.pt"), "w") as f:
        f.write("mock content")
        
    # 2. Test Listing
    gens = await list_generations()
    print("Found Generations:")
    found = False
    for g in gens:
        # print(g) # Remove print to avoid clutter, search specifically
        if g["id"] == "stage_99/gen_123":
            found = True
            print(f"SUCCESS: Found correct ID: {g['id']} | Name: {g['name']}")
            break
            
    if not found:
        print("FAILURE: Did not find stage_99/gen_123 in listing")
        print(f"Full List: {[x['id'] for x in gens]}")
        
    # 3. Test Loading Logic (Path Resolution)
    # We can't easily fully instantiate TrainingSession without lots of mocks, 
    # but we can verify the path resolution logic if we extract it or mock enough.
    # Let's just create a TrainingSession and mock the trainer.
    
    session = TrainingSession(MagicMock())
    session.trainer = MagicMock()
    session.trainer.config.envs_per_agent = 1 # avoid connection issues
    session.ensure_trainer = MagicMock()
    session._playback_loop = MagicMock()
    session._run_loop = MagicMock()
    session.thread = MagicMock()
    
    # Override standard libs to avoid actual torch load
    # We just want to see if it ATTEMPTS to load from correct path
    original_exists = os.path.exists
    original_isdir = os.path.isdir
    
    # We rely on actual filesystem for existence checks which we just created
    
    try:
        # Mock torch.load to avoid error
        import torch
        torch.load = MagicMock(return_value={})
        
        # Call start with our new ID
        print("Attempting to load 'stage_99/gen_123'...")
        session.start(model_name="stage_99/gen_123")
        
        # Verify log called with correct loading message
        # "Loading population from generation folder: stage_99/gen_123..."
        
        calls = session.trainer.log.call_args_list
        loaded = False
        for args, kwargs in calls:
            msg = args[0]
            if "Loading population from generation folder: stage_99/gen_123" in msg:
                 loaded = True
                 print("SUCCESS: Session attempted to load from correct path.")
                 
        if not loaded:
             print("FAILURE: Session did not log loading from expected path.")
             print("Logs:", [c[0][0] for c in calls])

    except Exception as e:
        print(f"Error during load test: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    shutil.rmtree(stage_dir)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(test_listing_and_loading())
