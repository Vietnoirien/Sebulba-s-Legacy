import multiprocessing as mp
import threading
import asyncio
import uuid
import random
import os
import queue
import torch
from training.ppo import PPOTrainer
from training.telemetry import TelemetryWorker
from training.curriculum.stages import STAGE_TEAM

class TrainingSession:
    def __init__(self, manager):
        self.manager = manager # WebSocket ConnectionManager
        self.trainer = None
        self.thread = None
        self.stop_event = threading.Event()
        self.running = False
        self.loop = None # Reference to main event loop
        self.playback_queue = asyncio.Queue(maxsize=5000)
        self.playback_task = None
        self.stats = {} 
        
        # Telemetry Offloading
        self.telemetry_in_queue = mp.Queue(maxsize=1000) # PPO -> Worker
        self.telemetry_out_queue = mp.Queue(maxsize=1000) # Worker -> Session
        self.telemetry_worker_stop = mp.Event()
        self.telemetry_process = None
        self.consumer_thread = None

    def _safe_load_state_dict(self, model, state_dict):
        """
         robustly loads state_dict handling the '_orig_mod.' prefix mismatch
        caused by torch.compile wrapping.
        """
        # Detect if model is compiled (OptimizedModule)
        # Note: OptimizedModule is not directly importable easily, check class name or structure
        is_compiled = model.__class__.__name__ == "OptimizedModule"
        
        # Check first key of state dict
        if not state_dict:
            return # Empty
            
        k0 = next(iter(state_dict.keys()))
        is_checkpoint_compiled = k0.startswith("_orig_mod.")
        
        new_state_dict = {}
        
        if is_compiled and not is_checkpoint_compiled:
            # Model needs prefix, Checkpoint lacks it -> Add prefix
            for k, v in state_dict.items():
                new_state_dict[f"_orig_mod.{k}"] = v
        elif not is_compiled and is_checkpoint_compiled:
            # Model lacks prefix, Checkpoint has it -> Remove prefix
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k.replace("_orig_mod.", "")] = v
                else:
                    new_state_dict[k] = v
        else:
            # Match (Both compiled or both raw)
            new_state_dict = state_dict

        # Load
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            # Fallback: Try strict=False if keys are still wonky, but log it
            print(f"Warning: Strict loading failed, trying strict=False. Error: {e}")
            model.load_state_dict(new_state_dict, strict=False)

    def set_loop(self, loop):
        self.loop = loop

    def _log_callback(self, msg):
        """Callback to receive logs from Trainer and broadcast to UI."""
        if self.loop and self.loop.is_running():
             payload = {
                 "type": "telemetry",
                 "step": 0, 
                 "match_id": self.trainer.match_id if self.trainer else "sys",
                 "stats": {},
                 "logs": [str(msg)],
                 "race_state": None
             }
             asyncio.run_coroutine_threadsafe(self.manager.broadcast(payload), self.loop)

    def ensure_trainer(self):
        if self.trainer is None:
            self.trainer = PPOTrainer(logger_callback=self._log_callback)
        return self.trainer

    def start(self, model_name=None, curriculum_mode="auto", curriculum_stage=0, config=None, max_checkpoints=5):
        if self.running:
            return
        
        self.stop_event.clear()
        self.telemetry_worker_stop.clear()
        self.running = True
        
        # Start Telemetry Worker
        print("Starting Telemetry Worker...")
        # Re-init if needed (assuming fresh start or restart)
        # Note: If reusing same session instance, queues persist.
        self.telemetry_process = TelemetryWorker(
            self.telemetry_in_queue, 
            self.telemetry_out_queue, 
            self.telemetry_worker_stop
        )
        self.telemetry_process.start()
        
        # Start Consumer Thread
        self.consumer_thread = threading.Thread(target=self._telemetry_consumer, daemon=True)
        self.consumer_thread.start()
        
        self.ensure_trainer()
            
        # Apply Curriculum Settings
        self.trainer.curriculum_mode = curriculum_mode
        self.trainer.env.curriculum_stage = curriculum_stage
        self.trainer.log(f"Session started with Curriculum: {self.trainer.curriculum_mode} | Stage: {self.trainer.env.curriculum_stage}")
        
        # Apply Max Checkpoints Config
        self.trainer.config.max_checkpoints_to_keep = max_checkpoints
        self.trainer.log(f"Config: Max Checkpoints to Keep = {max_checkpoints}")

        # [FIX] Sync Curriculum Manager and Env Config explicitly
        # If we just set env.curriculum_stage int, env.config remains stuck on previous default (Nursery).
        # We must force the manager to switch and apply config.
        if self.trainer.curriculum.current_stage_id != curriculum_stage:
             self.trainer.curriculum.set_stage(curriculum_stage)
             # Apply the config immediately to env
             stage_cfg = self.trainer.curriculum.current_stage.get_env_config()
             self.trainer.env.set_stage(curriculum_stage, stage_cfg)
             self.trainer.log(f"Explicitly synced Env Config for Stage {curriculum_stage} (Gen Type: {stage_cfg.track_gen_type})")

        # Apply Initial Config if provided (Fix for race condition)
        if config:
            self.trainer.log(f"Applying initial configuration...")
            self.update_config(config)
            
        # Load Model Logic
        if model_name and model_name != "scratch":
             # 1. Try to find in League Registry
             path = None
             if self.trainer.league:
                 entry = next((e for e in self.trainer.league.registry if e['id'] == model_name), None)
                 if entry:
                     path = entry['path']
             
             # 2. Fallback to default checkpoints dir
             if path is None:
                 if model_name.startswith("gen_") and "agent" in model_name: 
                     path = f"data/checkpoints/{model_name}.pt"
                 # Support new stage-based paths (e.g. stage_1/gen_25) passed as ID
                 elif os.path.isdir(f"data/{model_name}"):
                     path = f"data/{model_name}"
                 # Support legacy generations
                 elif os.path.isdir(f"data/generations/{model_name}"):
                     path = f"data/generations/{model_name}"
                 else:
                     path = f"data/checkpoints/{model_name}.pt"
             
             import torch
             
             # Case A: Load Full Generation (Population)
             if os.path.isdir(path):
                 self.trainer.log(f"Loading population from generation folder: {model_name}...")
                 loaded_count = 0
                 try:
                     for fname in os.listdir(path):
                         if fname.startswith("agent_") and fname.endswith(".pt"):
                             try:
                                 agent_id = int(fname.split("_")[1].split(".")[0])
                                 if 0 <= agent_id < len(self.trainer.population):
                                     agent_path = os.path.join(path, fname)
                                     state = torch.load(agent_path, map_location=self.trainer.device)
                                     self._safe_load_state_dict(self.trainer.population[agent_id]['agent'], state)
                                     loaded_count += 1
                             except Exception as e:
                                 self.trainer.log(f"Skipped {fname}: {e}")
                     
                     if loaded_count > 0:
                         self.trainer.log(f"Successfully loaded {loaded_count} agents from {model_name}.")
                         import re
                         match = re.search(r"gen_(\d+)", model_name)
                         if match:
                             gen_num = int(match.group(1))
                             self.trainer.generation = gen_num
                             self.trainer.log(f"Resumed from Generation {gen_num}")
                         
                         # Load RMS Stats
                         self._load_rms_stats(path)



                         # IMPORTANT: Sync to Vectorized Stack
                         self.trainer.sync_agents_to_vectorized()
                             
                     else:
                         self.trainer.log(f"No valid agent files found in {model_name}")
                         
                 except Exception as e:
                     self.trainer.log(f"Failed to load generation {model_name}: {e}")

             # Case B: Load Single Checkpoint
             elif os.path.exists(path):
                 self.trainer.log(f"Loading initial model: {model_name} from {path}")
                 try:
                    state = torch.load(path, map_location=self.trainer.device)
                    # Use broadcast to sync all
                    self.trainer.broadcast_checkpoint(state)
                    self.trainer.active_model_name = model_name
                    import re
                    match = re.search(r"gen_(\d+)", model_name)
                    if match:
                        gen_num = int(match.group(1))
                        self.trainer.generation = gen_num
                        self.trainer.log(f"Resumed from Generation {gen_num}")
                    
                    # Try to load RMS from parent dir (assuming standard structure data/generations/gen_X/agent_Y.pt)
                    parent_dir = os.path.dirname(path)
                    self._load_rms_stats(parent_dir)
                    
                 except Exception as e:
                     self.trainer.log(f"Failed to load {model_name}: {e}")
             else:
                 self.trainer.log(f"Model file or directory not found: {path} | {model_name}")
            
        # Start Playback Task correctly (Regardless of model loading)
        if self.loop:
            self.playback_task = asyncio.run_coroutine_threadsafe(self._playback_loop(), self.loop)

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _load_rms_stats(self, directory):
        """Helper to load normalization statistics if present."""
        rms_path = os.path.join(directory, "rms_stats.pt")
        if os.path.exists(rms_path):
            try:
                state = torch.load(rms_path, map_location=self.trainer.device)
                
                if 'self' in state:
                    self.trainer.rms_self.load_state_dict(state['self'])
                if 'ent' in state:
                    self.trainer.rms_ent.load_state_dict(state['ent'])
                if 'cp' in state:
                    self.trainer.rms_cp.load_state_dict(state['cp'])
                    
                self.trainer.log(f"Successfully loaded normalization statistics from {rms_path}")
            except Exception as e:
                self.trainer.log(f"Failed to load RMS stats from {rms_path}: {e}")
        else:
            self.trainer.log(f"Warning: No normalization statistics found at {rms_path}. Agents may perform poorly.")



    def stop(self):
        if not self.running:
            return
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.playback_task:
            self.playback_task.cancel()
            
        self.telemetry_worker_stop.set()
        if self.telemetry_process and self.telemetry_process.is_alive():
            self.telemetry_process.join(timeout=2.0)
            if self.telemetry_process.is_alive():
                self.telemetry_process.terminate()
                
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=1.0)
            
        while not self.telemetry_in_queue.empty():
            try: self.telemetry_in_queue.get_nowait()
            except: break
            
        # Explicitly close queues to release underlying pipes
        try:
             self.telemetry_in_queue.cancel_join_thread()
             self.telemetry_in_queue.close()
             # self.telemetry_in_queue.join_thread() # Avoid joining, preventing hang if reader died
        except: pass
        
        try:
             self.telemetry_out_queue.cancel_join_thread()
             self.telemetry_out_queue.close()
             # self.telemetry_out_queue.join_thread()
        except: pass

        self.running = False

    def reset(self):
        self.stop()
        self.trainer = None
        self.stats = {}
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.telemetry_in_queue = mp.Queue(maxsize=1000)
        self.telemetry_out_queue = mp.Queue(maxsize=1000)

    def update_config(self, config):
        if self.trainer:
            if "rewards" in config:
                 new_rewards = config["rewards"]
                 if "weights" in new_rewards:
                     self.trainer.reward_config["weights"].update(new_rewards["weights"])
                     # Also update the tensor!
                     # Optimization: For now we just update weights dict, 
                     # but in PBT we need to update the tensor row for the leader?
                     # Or all agents?
                     # If we are "tuning parameters", usually we want to override the whole population 
                     # or at least the leader.
                     # Let's broadcast to all agents for now to make the UI effective.
                     for i, p in enumerate(self.trainer.population):
                         for k, v in new_rewards["weights"].items():
                             k_int = int(k)
                             p['weights'][k_int] = float(v)
                             # Update tensor
                             start_idx = i * (self.trainer.config.num_envs // len(self.trainer.population))
                             end_idx = start_idx + (self.trainer.config.num_envs // len(self.trainer.population))
                             self.trainer.reward_weights_tensor[start_idx:end_idx, k_int] = float(v)

                 if "tau" in new_rewards:
                     self.trainer.reward_config["tau"] = float(new_rewards["tau"])
                 if "beta" in new_rewards: # Team Spirit alias?
                     # beta usually refers to entropy or similar, but here maybe team_spirit?
                     # Config.py doesn't have BETA. 
                     # self.trainer.team_spirit is what we want.
                     pass 
                 if "team_spirit" in new_rewards:
                     self.trainer.team_spirit = float(new_rewards["team_spirit"])
                     
            if "curriculum" in config:
                curr = config["curriculum"]
                if "stage" in curr:
                    self.trainer.env.curriculum_stage = int(curr["stage"])
                if "difficulty" in curr:
                    self.trainer.env.bot_difficulty = float(curr["difficulty"])
                    
            if "transitions" in config:
                trans = config["transitions"]
                # Update known keys
                for k in ["solo_efficiency_threshold", "solo_consistency_threshold",
                          "duel_graduation_difficulty", "duel_graduation_win_rate", "duel_graduation_checks",
                          "duel_graduation_difficulty", "duel_graduation_win_rate", "duel_graduation_checks",
                          "duel_graduation_denial_rate", "duel_graduation_blocker_impact",
                          "team_graduation_difficulty", "team_graduation_win_rate", "team_graduation_checks"]:
                     if k in trans:
                         setattr(self.trainer.curriculum_config, k, float(trans[k]))
                
            print(f"Trainer Config Updated: {self.trainer.reward_config}")
            
            if "bot_config" in config:
                bot_cfg = config["bot_config"]
                for k in ["intercept_offset_scale", "ramming_speed_scale", "difficulty_noise_scale", "thrust_scale"]:
                     if k in bot_cfg:
                         if hasattr(self.trainer.env, "bot_config"):
                             setattr(self.trainer.env.bot_config, k, float(bot_cfg[k]))
                print(f"Bot Config Updated: {self.trainer.env.bot_config}")

    async def _playback_loop(self):
        print("Starting Playback Loop...")
        active_race = []
        while self.running:
            try:
                # If no active race, get one from buffer
                if not active_race:
                    active_race = await self.playback_queue.get()
                
                # Handling Generic Payloads (e.g. Full Replay)
                if isinstance(active_race, dict) and active_race.get("type") == "race_replay":
                     # Broadcast Full Replay
                     await self.manager.broadcast(active_race)
                     
                     # Pace the stream (Wait for duration of race)
                     # Assuming 20 FPS recording
                     await asyncio.sleep(0.016) # Cap at ~60Hz to prevent flooding frontend
                     
                     active_race = [] # Done
                     continue

                # Handling Legacy List-of-Frames (if reverted or mixed)
                elif isinstance(active_race, list):
                    payload = active_race.pop(0)
                    if self.stats:
                        payload["stats"].update(self.stats)
                    await self.manager.broadcast(payload)
                    await asyncio.sleep(0.016) # Cap at ~60Hz
                    if not active_race:
                        active_race = [] # Clear if empty
                
                else:
                    # Garbage collection for unknown types preventing infinite loop
                    print(f"WARNING: Unknown object in playback queue: {type(active_race)}")
                    active_race = []
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Playback Error: {e}")
                active_race = [] # Clear if error
                await asyncio.sleep(1.0)

    def _telemetry_consumer(self):
        print("DEBUG: Telemetry Consumer Started")
        while self.running:
            try:
                try:
                    msg = self.telemetry_out_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                msg_type = msg.get("type")
                
                if msg_type == "telemetry_stats":
                     stats_payload = {
                        "type": "telemetry",
                        "step": 0, 
                        "match_id": self.trainer.match_id if self.trainer else "sys",
                        "stats": msg["payload"],
                        "logs": [],
                        "race_state": None
                     }
                     if self.loop:
                         asyncio.run_coroutine_threadsafe(self.manager.broadcast(stats_payload), self.loop)
                         
                elif msg_type == "race_replay":
                    race_params = msg["payload"]
                    
                    if self.loop:
                        # Thread-Safe Queue Push
                        # Use call_soon_threadsafe because we are invalidating the GIL/Asyncio boundary
                        self.loop.call_soon_threadsafe(self.playback_queue.put_nowait, msg)
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Consumer Error: {e}")

    def _telemetry_callback(self, step, sps, fps_train, reward, win_rate, env_idx, loss, log_line=None, is_done=False, step_rewards=None, step_actions=None, league_stats=None, collision_flags=None):
        if not self.running:
            return

        try:
            if log_line:
                stats_data = {
                    "fps_physics": float(sps),
                    "fps_training": float(fps_train),
                    "reward_mean": float(reward),
                    "loss": float(loss),
                    "win_rate": float(win_rate),
                    "active_model": "training",
                    "generation": getattr(self.trainer, 'generation', 0),
                    "curriculum_stage": int(self.trainer.env.curriculum_stage) if self.trainer else 0,
                    "league": league_stats
                }
                try:
                    self.telemetry_in_queue.put_nowait({
                        "type": "stats",
                        "payload": stats_data
                    })
                except queue.Full:
                    pass
                return

            # Fetch Data
            t_env = self.trainer.env
            idx = int(env_idx)
            pos = t_env.physics.pos[idx].detach().cpu().numpy() # [4, 2]
            vel = t_env.physics.vel[idx].detach().cpu().numpy()
            angle = t_env.physics.angle[idx].detach().cpu().numpy()
            laps = t_env.laps[idx].detach().cpu().numpy()
            next_cps = t_env.next_cp_id[idx].detach().cpu().numpy()
            
            # Slice checkpoints to valid count
            n_cp = t_env.num_checkpoints[idx].item()
            if n_cp == 0:
                # Fallback: Send all checkpoints (Stage 0 usually has fixed count)
                # print(f"WARNING: n_cp is 0 for env {idx}. Sending all 8.") # Reduce spam 
                checkpoints = t_env.checkpoints[idx].detach().cpu().numpy()
            else:
                checkpoints = t_env.checkpoints[idx, :n_cp].detach().cpu().numpy() # [n_cp, 2]
            
            # Identify Agent (assuming standard PBT config)
            # We can't easily import ENVS_PER_AGENT here due to circular deps if we aren't careful, 
            # but we know it's NUM_ENVS // 32 usually.
            # Best to just pass env_idx and let frontend handle or pass helper.
            # Fix: Use trainer config
            agent_id = idx // self.trainer.config.envs_per_agent
            
            # Check Pareto Status
            is_pareto = False
            if hasattr(self.trainer, 'pareto_indices'):
                if agent_id in self.trainer.pareto_indices:
                    is_pareto = True

            payload = {
                "type": "step",
                "step": step,
                "env_idx": idx,
                "agent_id": agent_id,
                "is_pareto": is_pareto,
                "generation": self.trainer.generation,
                "iteration": self.trainer.iteration,
                "dones": bool(is_done),
                "rewards": step_rewards,
                "actions": step_actions,
                "match_id": self.trainer.match_id,
                "pos": pos,
                "vel": vel,
                "angle": angle,
                "laps": laps,
                "next_cps": next_cps,
                "checkpoints": checkpoints,
                "collision_flags": collision_flags,
                "ranks": t_env.prev_ranks[idx].detach().cpu().numpy() if hasattr(t_env, 'prev_ranks') else None
            }
            
            try:
                self.telemetry_in_queue.put_nowait(payload)
            except queue.Full:
                print("WARNING: Telemetry Queue FULL. Dropping frame.")
                pass

        except Exception as e:
            # Ignore queue closed errors during shutdown
            if isinstance(e, ValueError) and "closed" in str(e):
                return
                
            import traceback
            traceback.print_exc()
            print(f"Callback Error: {e}")

    def _run_loop(self):
        try:
            print("Training Thread Started")
            self.trainer.train_loop(
                stop_event=self.stop_event, 
                telemetry_callback=self._telemetry_callback
            )
            print("Training Thread Finished Gracefully")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"CRITICAL: Training Thread Crashed: {e}")
