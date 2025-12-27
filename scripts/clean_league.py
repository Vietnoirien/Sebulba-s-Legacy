import json
import os
import glob

LEAGUE_FILE = "data/league.json"

def clean_league():
    if not os.path.exists(LEAGUE_FILE):
        print(f"League file {LEAGUE_FILE} not found.")
        return

    try:
        with open(LEAGUE_FILE, "r") as f:
            registry = json.load(f)
    except Exception as e:
        print(f"Error reading league file: {e}")
        registry = []

    print(f"Found {len(registry)} entries in league.")

    deleted_count = 0
    for entry in registry:
        path = entry.get('path')
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"Deleted checkpoint: {path}")
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
        elif path:
             print(f"Checkpoint not found (already deleted?): {path}")

    # Also clean up any orphan gen_* files in data/checkpoints if they match pattern?
    # Safety: Only delete what's in registry for now to avoid deleting manual saves.
    
    # Reset Registry
    with open(LEAGUE_FILE, "w") as f:
        json.dump([], f)
        
    print(f"League registry reset. Deleted {deleted_count} files.")

if __name__ == "__main__":
    clean_league()
