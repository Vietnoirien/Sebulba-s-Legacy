import os
import json
import torch
import random
import glob
from pathlib import Path

LEAGUE_DIR = "data/checkpoints"
LEAGUE_FILE = "data/league.json"

class LeagueManager:
    def __init__(self):
        # Ensure dirs
        os.makedirs(LEAGUE_DIR, exist_ok=True)
        self.league_path = Path(LEAGUE_FILE)
        
        self.registry = []
        self._load_registry()
        
    def _load_registry(self):
        if self.league_path.exists():
            with open(self.league_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = []
            
    def _save_registry(self):
        with open(self.league_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def save_checkpoint(self, agent, step, name=None):
        if name is None:
            name = f"model_{step}"
            
        filename = os.path.join(LEAGUE_DIR, f"{name}.pt")
        torch.save(agent.state_dict(), filename)
        
        # Add to registry
        entry = {
            "id": name,
            "path": filename,
            "step": step,
            "elo": 1000,
            "matches": 0,
            "creation_time": 0 # TODO time
        }
        self.registry.append(entry)
        self._save_registry()

    def register_agent(self, name, path, step, metrics=None):
        """
        Register an existing checkpoint file to the league.
        """
        if not os.path.exists(path):
            print(f"Warning: Trying to register non-existent path {path}")
            return

        entry = {
            "id": name,
            "path": path,
            "step": step,
            "elo": 1000,
            "matches": 0,
            "creation_time": 0, # TODO time
            "metrics": metrics or {}
        }
        
        # Check if exists
        if any(e['id'] == name for e in self.registry):
             # Update?
             pass
        else:
             self.registry.append(entry)
             self._save_registry()

        # Pruning (Hall of Fame) logic
        if len(self.registry) > 100:
            self._prune()
        
            
    def _prune(self):
        # Sort by ELO
        sorted_reg = sorted(self.registry, key=lambda x: x['elo'], reverse=True)
        # Keep top 10 (Anchors)
        anchors = sorted_reg[:10]
        candidates = sorted_reg[10:]
        
        # Remove oldest from candidates
        candidates.sort(key=lambda x: x['step']) # Sort by age
        to_remove = candidates[0]
        
        # Delete file
        if os.path.exists(to_remove['path']):
            os.remove(to_remove['path'])
            
        # Update registry
        self.registry = [e for e in self.registry if e['id'] != to_remove['id']]
        self._save_registry()

    def sample_opponent(self):
        """
        Returns a path to a checkpoint.
        Strategy: 80% Latest (Self), 20% Historical.
        """
        if not self.registry or random.random() < 0.8:
            return None # None implies current policy
            
        # Sample from History
        entry = random.choice(self.registry)
        return entry['path']

    def update_elo(self, id_a, id_b, score_a):
        """
        id_a: Player A ID
        id_b: Player B ID
        score_a: 1.0 (Win), 0.5 (Draw), 0.0 (Loss)
        """
        # Find entries
        entry_a = next((e for e in self.registry if e['id'] == id_a), None)
        entry_b = next((e for e in self.registry if e['id'] == id_b), None)
        
        if not entry_a or not entry_b:
            return
            
        ra = entry_a['elo']
        rb = entry_b['elo']
        
        # Expected score
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        
        # K-Factor
        k = 32
        
        # Update
        new_ra = ra + k * (score_a - ea)
        new_rb = rb + k * ((1.0 - score_a) - (1.0 - ea))
        
        entry_a['elo'] = int(new_ra)
        entry_b['elo'] = int(new_rb)
        
        entry_a['matches'] += 1
        entry_b['matches'] += 1
        
        self._save_registry()

    def remove_opponent(self, path):
        """Remove a checkpoint from the league registry and filesystem."""
        # Find entry
        entry = next((e for e in self.registry if e['path'] == path), None)
        
        if entry:
            self.registry.remove(entry)
            self._save_registry()
            print(f"Removed {path} from League Registry.")
            
        # Remove file if it exists
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Deleted file: {path}")
            except OSError as e:
                print(f"Error deleting {path}: {e}")

