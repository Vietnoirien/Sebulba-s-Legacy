import multiprocessing as mp
import time
import queue
import numpy as np
import struct
import base64
import uuid

class TelemetryWorker(mp.Process):
    """
    Dedicated Worker Process for handling Telemetry Data.
    Consumes raw numpy headers from PPO Loop.
    Manages Recording Buffers.
    Produces formatted payloads for the WebSocket Manager.
    """
    def __init__(self, input_queue, output_queue, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        
        # Internal Recording Buffers
        # Dict[env_idx, List[bytes]]
        self.buffers = {}
        
        # Stats Cache
        self.stats = {}

    def run(self):
        # Ignore SIGINT in worker to let parent handle cleanup
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        while not self.stop_event.is_set():
            try:
                try:
                    data = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                msg_type = data.get("type")
                
                if msg_type == "stats":
                    # Update internal stats cache
                    self.stats.update(data["payload"])
                    self.output_queue.put({
                        "type": "telemetry_stats",
                        "payload": data["payload"]
                    })
                    
                elif msg_type == "step":
                    self.process_step(data)
                    
            except Exception as e:
                print(f"Telemetry Worker Error: {e}")
                time.sleep(0.1)

    def process_step(self, data):
        env_idx = data['env_idx']
        is_done = data['dones']
        
        # Init buffer struct if needed
        if env_idx not in self.buffers:
            self.buffers[env_idx] = {
                "frames": [],
                "race_id": str(uuid.uuid4()) # Generate Race ID on first contact
            }
            
        buffer_obj = self.buffers[env_idx]
        
        # ALWAYS Pack Frame (Throttling is handled by PPO upstream at 20Hz)
        frame_bytes = self.pack_frame(data)
        buffer_obj["frames"].append(frame_bytes)
        
        # CHUNKED STREAMING
        # If buffer size > Threshold (e.g. 50 frames ~ 100-200 bytes * 50 = 10KB)
        if len(buffer_obj["frames"]) >= 50:
             self.flush_chunk(env_idx, is_final=False, data_ref=data)

        if is_done:
            # Flush remaining frames as final chunk
            self.flush_chunk(env_idx, is_final=True, data_ref=data)
            # Clean up
            del self.buffers[env_idx]

    def flush_chunk(self, env_idx, is_final, data_ref):
        if env_idx not in self.buffers: return
        
        buffer_obj = self.buffers[env_idx]
        frames = buffer_obj["frames"]
        
        if not frames and not is_final:
            return

        chunk_data = b''.join(frames)
        b64_data = base64.b64encode(chunk_data).decode('ascii')
        
        # Checkpoints: Send ONLY with the first chunk of a race?
        # Or safely send always? Front end can ignore. 
        # Sending always is safer for "late joiners" but wastes BW.
        # Let's send always for now (small overhead).
        
        cps = []
        if 'checkpoints' in data_ref:
            for i, cp in enumerate(data_ref['checkpoints']):
                 cps.append({"x": float(cp[0]), "y": float(cp[1]), "id": i, "radius": 600})
        
        payload = {
            "type": "race_replay",
            "format": "binary_base64",
            "payload": b64_data,
            "checkpoints": cps,
            "frame_count": len(frames),
            # Metadata
            "env_idx": env_idx,
            "race_id": buffer_obj["race_id"], # Unique Race ID
            "agent_id": data_ref.get("agent_id", 0),
            "generation": data_ref.get("generation", 0),
            "iteration": data_ref.get("iteration", 0),
            "is_partial": True,
            "is_final": is_final
        }
        
        self.output_queue.put(payload)
        
        # Clear buffer
        buffer_obj["frames"] = []

    def pack_frame(self, data):
        """
        Pack frame data into binary struct (148 bytes total)
        Header (16B): Magic(2) Type(1) Step(4) EnvIdx(2) Reserved(7)
        Pod (33B) * 4: X(4) Y(4) VX(4) VY(4) Ang(4) Thrust(4) Reward(4) Lap(2) NextCP(2)
        Actually 4*32 = 128. Total 144.
        """
        step = int(data['step'])
        env_idx = int(data['env_idx'])
        
        # Header: Magic=0xDEAD, Type=1 (Frame)
        header = struct.pack('<HBIH7x', 0xDEAD, 1, step, env_idx)
        
        # Pods
        pos = data['pos']       
        vel = data['vel']       
        angle = data['angle']   
        laps = data['laps']     
        next_cps = data['next_cps'] 
        rewards = data['rewards'] 
        actions = data['actions'] 
        
        pods_bytes = []
        # Expect collision_flags in data if available
        coll_flags = data.get("collision_flags") # [4] numpy array or None

        for i in range(4):
            team = i // 2
            p_thrust = float(actions[i, 0]) * 100.0 if actions is not None else 0.0
            p_shield = float(actions[i, 2]) if actions is not None else 0.0
            p_boost = float(actions[i, 3]) if actions is not None else 0.0
            p_reward = float(rewards[team]) if rewards is not None else 0.0
            
            p_collision = 0.0
            if coll_flags is not None:
                p_collision = float(coll_flags[i])

            # 10 floats (40 bytes) + 2 shorts (4 bytes) = 44 bytes
            # Added p_collision at end of floats
            p_data = struct.pack('<10f2H', 
                float(pos[i, 0]), float(pos[i, 1]),
                float(vel[i, 0]), float(vel[i, 1]),
                float(angle[i]),
                p_thrust,
                p_shield,
                p_boost,
                p_reward,
                p_collision,
                int(laps[i]),
                int(next_cps[i])
            )
            pods_bytes.append(p_data)
            
        return header + b''.join(pods_bytes)
