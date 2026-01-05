
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import uuid
import math # Added math import
from typing import Dict, List, Any, Optional # Added Optional to typing
from pydantic import BaseModel
import os # Added os
from training.session import TrainingSession # Added TrainingSession import
from training.ppo import PPOTrainer # Needed for lazy init? Or accessing trainer type
import torch # Needed for torch.load
from fastapi.responses import FileResponse
from export import export_model


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    loop = asyncio.get_event_loop()
    # session is defined later but available at runtime
    session.set_loop(loop)
    yield
    # Shutdown
    print("Shutting down session...")
    session.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TelemetryData(BaseModel):
    fps_physics: float
    fps_training: float
    reward_mean: float
    loss: float
    win_rate: float
    active_model: str
    curriculum_stage: int
    league: Optional[Dict[str, Any]] = None

class RaceState(BaseModel):
    pods: List[Dict[str, Any]]
    checkpoints: List[Dict[str, Any]]

class TelemetryMessage(BaseModel):
    type: str = "telemetry"
    step: int
    match_id: str
    stats: TelemetryData
    logs: List[str] = []
    race_state: RaceState

class HasName(BaseModel):
    name: str
    data: Optional[Dict[str, Any]] = None


# --- Config Management ---
class ConfigItem(BaseModel):
    name: str
    config: Dict[str, Any]

class ConfigList(BaseModel):
    configs: List[HasName]

# --- Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Convert to JSON string
        json_str = json.dumps(message)
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json_str)
            except Exception:
                # Failed to send (Connection closed), mark for removal
                to_remove.append(connection)
        
        for conn in to_remove:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client (Control)
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                # Handle commands
                if payload.get("type") == "command":
                    action = payload.get("action")
                    print(f"Received command: {action}")
                    # Dispatch command (To be implemented with Training Manager)
                
                elif payload.get("type") == "config":
                    config = payload.get("payload")
                    print(f"Received config update: {config}")
                    session.update_config(config)
                    
            except json.JSONDecodeError:
                pass
            
            # Temporary: Echo back or Ack? 
            # Usually the server broadcasts telemetry independently of client requests.
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WS Error: {e}")
        manager.disconnect(websocket)


# Session Manager
session = TrainingSession(manager)


class StartPayload(BaseModel):
    model: Optional[str] = "scratch"
    curriculum_mode: Optional[str] = "auto"
    curriculum_stage: Optional[int] = 0
    config: Optional[Dict[str, Any]] = None

@app.post("/api/start")
async def start_training(payload: StartPayload = None):
    model_name = "scratch"
    curr_mode = "auto"
    curr_stage = 0
    initial_config = None
    
    if payload:
        model_name = payload.model
        curr_mode = payload.curriculum_mode
        curr_stage = payload.curriculum_stage
        initial_config = payload.config

    try:
        session.start(model_name=model_name, curriculum_mode=curr_mode, curriculum_stage=curr_stage, config=initial_config)
        return {"status": "started", "model": model_name, "curriculum": curr_mode, "stage": curr_stage}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/api/stop")
async def stop_training():
    session.stop()
    return {"status": "stopped"}

@app.post("/api/reset")
async def reset_training():
    session.reset()
    return {"status": "reset"}

@app.delete("/api/checkpoints/reset")
async def wipe_checkpoints():
    """
    Deletes all .pt files in data/checkpoints/ and data/generations/ and clears the league registry.
    """
    import shutil
    try:
        # 1. Delete Checkpoints (Deep Clean)
        folder = "data/checkpoints"
        if os.path.exists(folder):
            try:
                # Remove entire folder and recreate to ensure no stale files
                shutil.rmtree(folder)
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                print(f"Failed to clear checkpoints: {e}")

        # 2. Delete Generations (Deep Clean)
        gen_folder = "data/generations"
        if os.path.exists(gen_folder):
            try:
                shutil.rmtree(gen_folder)
                os.makedirs(gen_folder, exist_ok=True)
            except Exception as e:
                print(f"Failed to delete generations: {e}")
        
        # 3. Clear Registry & Payoff (In-Memory)
        if session.trainer and session.trainer.league:
            session.trainer.league.registry = []
            session.trainer.league.payoff = {} # Reset Match History
            session.trainer.league._save_registry()
            session.trainer.league._save_payoff()
            
        # 4. Clear Registry & Payoff (Filesystem)
        # Even if trainer is active, force wipe files to be sure
        if os.path.exists("data/league.json"):
            with open("data/league.json", "w") as f:
                json.dump([], f)
        
        if os.path.exists("data/payoff.json"):
            try:
                os.remove("data/payoff.json")
            except:
                pass
                
        return {"status": "wiped", "message": "All checkpoints, generations, and league history wiped."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/status")
async def get_status():
    return {"running": session.running}

# --- Checkpoint Management ---
@app.get("/api/checkpoints")
async def list_checkpoints():
    # Helper to load directly from file
    def load_league_json():
        if os.path.exists("data/league.json"):
            try:
                with open("data/league.json", "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    # 1. Use Trainer Registry if available and populated
    if session.trainer and session.trainer.league and session.trainer.league.registry:
        return session.trainer.league.registry
        
    # 2. Fallback to file reading (Startup or after reset)
    return load_league_json()

@app.get("/api/generations")
async def list_generations():
    """
    Lists available generation folders in data/generations.
    Returns: List[Dict] with id, name, path, and stats if available.
    """
    generations = []
    base_dir = "data/generations"
    if not os.path.exists(base_dir):
        return []
        
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("gen_"):
            # Try to get timestamp
            try:
                ctime = os.path.getctime(path)
                
                # Count agents
                agents = [f for f in os.listdir(path) if f.endswith(".pt")]
                
                # ID is the folder name
                generations.append({
                    "id": name,
                    "name": name.replace("_", " ").title(), # gen_0 -> Gen 0
                    "path": path,
                    "creation_time": ctime,
                    "agent_count": len(agents)
                })
            except Exception:
                pass
                
    # Sort by creation time desc
    generations.sort(key=lambda x: x["creation_time"], reverse=True)
    return generations

@app.post("/api/checkpoints/{name}/load")
async def load_checkpoint(name: str):
    # Load into Trainer
    if not session.trainer:
         session.ensure_trainer() # Init if needed
    
    path = f"data/checkpoints/{name}.pt"
    if not os.path.exists(path):
        return {"error": "Checkpoint not found"}
        
    try:
        state = torch.load(path, map_location=session.trainer.device)
        session.trainer.broadcast_checkpoint(state)
        # Set Active Model Name
        session.trainer.active_model_name = name
        return {"status": "loaded", "name": name}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/checkpoints/{name}")
async def delete_checkpoint(name: str):
    path = f"data/checkpoints/{name}.pt"
    if os.path.exists(path):
        os.remove(path)
        # Update Registry
        if session.trainer:
             session.trainer.league.registry = [e for e in session.trainer.league.registry if e['id'] != name]
             session.trainer.league._save_registry()
        return {"status": "deleted"}
    return {"error": "Not found"}




@app.post("/api/export")
async def export_submission():
    # Determine model path
    # Use active model from session if available, else look for latest file
    model_name = "model_latest"
    if session.trainer and session.trainer.active_model_name:
        model_name = session.trainer.active_model_name
        
    model_path = f"data/checkpoints/{model_name}.pt"
    
    # If explicit model not found, find ANY .pt file to export (for testing)
    if not os.path.exists(model_path):
        if not os.path.exists("data/checkpoints"):
             return {"error": "No checkpoints directory found"}
        files = [f for f in os.listdir("data/checkpoints") if f.endswith(".pt")]
        if not files:
            return {"error": "No model found to export"}
        # Pick newest
        model_path = os.path.join("data/checkpoints", max(files, key=lambda x: os.path.getctime(os.path.join("data/checkpoints", x))))

    out_file = "submission.py"
    try:
        # Run export in thread pool if slow? It's fast enough.
        export_model(model_path, out_file)
        return FileResponse(out_file, media_type='application/x-python-code', filename="submission.py")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# --- Configuration Presets ---

DEFAULT_CONFIG = {
    "rewards": {
         "weights": {
             "0": 10000.0, "1": 5000.0, "2": 2000.0, "3": 50.0, "4": 0.2,
             "5": 0.5, "6": 5.0, "7": 10.0, "8": 1.0, "9": 10.0, "10": 2.0,
             "11": 5.0, "12": 10.0, "13": 500.0
         },
         "tau": 0.0,
         "team_spirit": 0.0
    },
    "curriculum": {"stage": 0, "difficulty": 0.0},
    "hyperparameters": {"lr": 1e-4, "ent_coef": 0.01},
    "transitions": {
         "solo_efficiency_threshold": 30.0,
         "solo_consistency_threshold": 2000.0,
         "duel_consistency_wr": 0.82,
         "duel_absolute_wr": 0.84,
         "duel_consistency_checks": 5,
         "team_consistency_wr": 0.85,
         "team_absolute_wr": 0.88,
         "team_consistency_checks": 5
    }
}

@app.get("/api/configs")
async def list_configs():
    """List all saved configuration presets."""
    configs = []
    
    # Always include Default
    configs.append({
        "name": "default",
        "data": DEFAULT_CONFIG
    })

    folder = "data/configs"
    if not os.path.exists(folder):
        return configs
        
    for f in os.listdir(folder):
        if f.endswith(".json"):
            try:
                with open(os.path.join(folder, f), "r") as file:
                    data = json.load(file)
                    configs.append({
                        "name": f.replace(".json", ""),
                        "data": data
                    })
            except Exception:
                pass
    
    return configs

@app.post("/api/configs")
async def save_config(item: ConfigItem):
    """Save a configuration preset."""
    folder = "data/configs"
    os.makedirs(folder, exist_ok=True)
    
    # Sanitize name
    safe_name = "".join([c for c in item.name if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_name:
        return {"error": "Invalid name"}
    
    if safe_name.lower() == "default":
        return {"error": "Cannot overwrite default preset"}
        
    path = os.path.join(folder, f"{safe_name}.json")
    try:
        with open(path, "w") as f:
            json.dump(item.config, f, indent=2)
        return {"status": "saved", "name": safe_name}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/configs/{name}")
async def delete_config(name: str):
    """Delete a configuration preset."""
    if name.lower() == "default":
        return {"error": "Cannot delete default preset"}
        
    folder = "data/configs"
    path = os.path.join(folder, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
    return {"error": "Not found"}

@app.post("/api/configs/load")
async def load_config_to_session(item: ConfigItem):
    """Apply a config directly to the running session."""
    try:
        session.update_config(item.config)
        return {"status": "applied"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
