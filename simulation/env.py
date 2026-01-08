import torch
import math
from simulation.gpu_physics import GPUPhysics
from simulation.tracks import TrackGenerator, START_POINT_MULT
from config import *

# Reward Indices and Weights imported from config


class PodRacerEnv:
    def __init__(self, num_envs, device='cuda', start_stage=STAGE_SOLO):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.physics = GPUPhysics(num_envs, device=device)
        self.curriculum_stage = start_stage 
        
        # Default Config (Solo)
        self.config = EnvConfig(
             mode_name="solo",
             track_gen_type="max_entropy",
             active_pods=[0],
             use_bots=False,
             step_penalty_active_pods=[0],
             orientation_active_pods=[0]
        )
        
        # New Configs
        self.bot_config = BotConfig()
        self.spawn_config = SpawnConfig()
        self.reward_scaling_config = RewardScalingConfig()

        # Map Mode Cache
        self.using_nursery_map = False
        self._update_map_mode()
        
        # Predefined Map State
        self.is_predefined_map = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        # Game State
        self.next_cp_id = torch.ones((num_envs, 4), dtype=torch.long, device=self.device) # Start at 1 (0 is start)
        self.laps = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device)
        self.timeouts = torch.full((num_envs, 4), TIMEOUT_STEPS, dtype=torch.long, device=self.device)
        self.cps_passed = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device)
        
        # Checkpoints [Batch, MaxCP, 2]
        self.checkpoints = torch.zeros((num_envs, MAX_CHECKPOINTS, 2), device=self.device)
        self.num_checkpoints = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        
        # Track steps since last CP for efficiency metric
        self.steps_last_cp = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device) # Track steps per CP
        
        # Metrics per env
        self.stage_metrics = {
            "solo_completes": 0,
            "solo_steps": 0,
            "checkpoint_hits": 0,
            "duel_wins": 0,
            "duel_games": 0,
            "recent_wins": 0,
            "recent_games": 0,
            "recent_episodes": 0,
            "blocker_impact": 0,
            "recent_denials": 0
        }
        
        # Rewards / Done
        self.rewards = torch.zeros((num_envs, 4), device=self.device) # Pod 0, 1, 2, 3
        self.dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self.winners = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        
        # Roles
        self.is_runner = torch.zeros((num_envs, 4), dtype=torch.bool, device=self.device)

        
        # Progress tracking for dense rewards
        self.prev_dist = torch.zeros((num_envs, 4), device=self.device)
        
        self.bot_difficulty = 0.0 # 0.0 to 1.0
        
        # Role Hysteresis
        self.role_lock_timer = torch.zeros((num_envs,), dtype=torch.long, device=self.device)


        
        # Rank Tracking for Potential-Based Reward
        self.prev_ranks = torch.zeros((num_envs, 4), dtype=torch.long, device=self.device)
        self.rank_range_cache = torch.arange(4, device=self.device).unsqueeze(0).expand(num_envs, 4)
        
        # --- OPTIMIZATION: Cache Static Indices for get_obs ---
        # 1. Batch Index [B, 4]
        self.cache_batch_idx = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(-1, 4)
        
        # 2. Team Indices [B, 4]
        # Pods 0,1 -> Team 0. Pods 2,3 -> Team 1.
        self.cache_team_idx = torch.tensor([0, 0, 1, 1], device=self.device).expand(num_envs, 4)
        
        # 3. Entity "Other" Indices [4, 3] -> [B, 4, 3] (Broadcast compatible)
        # 0: 1, 2, 3
        # 1: 0, 2, 3
        # 2: 0, 1, 3
        # 3: 0, 1, 2
        self.cache_other_indices = torch.tensor([
            [1, 2, 3],
            [0, 2, 3],
            [3, 0, 1],
            [2, 0, 1]
        ], device=self.device)
        
        # 4. Mate Team Check [B, 4, 3, 1]
        # p_team [4, 1] -> [0, 0, 1, 1]
        p_team = torch.tensor([0, 0, 1, 1], device=self.device).unsqueeze(1)
        # o_team [4, 3]
        o_team = self.cache_other_indices // 2
        # Compare [4, 1] vs [4, 3] -> [4, 3]
        # Expand to Batch [B, 4, 3, 1]
        self.cache_is_mate = (o_team == p_team).float().unsqueeze(0).unsqueeze(-1).expand(num_envs, -1, -1, -1)
        
        # 5. Checkpoint Batch Index for Entity/NextCP gather
        # self.checkpoints is [B, N, 2].
        # We need [B, 4, 3] indices for batch dim.
        self.cache_batch_idx_entity = torch.arange(num_envs, device=self.device).view(-1, 1, 1).expand(-1, 4, 3)

        self.reset()
        
    def _get_ranks(self, env_ids):
        """
        Calculate current rank (0-3) for each pod based on progress score.
        Score = Laps * 10000 + NextCP * 100 + (1 - Dist/20000)
        """
        # Gather data
        laps = self.laps[env_ids] # [N, 4]
        next_cp = self.next_cp_id[env_ids] # [N, 4]
        # Use prev_dist as it is updated every step/reset and represents 'dist to next cp'
        dists = self.prev_dist[env_ids] # [N, 4]
        
        # Max map dist approx 20000.
        dist_score = 1.0 - (dists / 20000.0)
        
        # FIX: Handle Final Leg (NextCP == 0)
        # NextCP=0 means passing last CP and aiming for Finish (Lap Completion).
        # We should count this as "NumCPs" progress, not 0.
        # Since NextCP starts at 1, 0 is unique to this state.
        effective_cp = next_cp.clone()
        # Create mask for envs where next_cp is 0
        mask_finish = (effective_cp == 0)
        
        # We need num_checkpoints for each env
        # self.num_checkpoints is [N_envs]
        # We need to broadcast or index correctly
        # env_ids is passed in, so we look up specific envs
        n_cps_local = self.num_checkpoints[env_ids]
        n_cps_expanded = n_cps_local.unsqueeze(1).expand(-1, 4)
        
        # Optimized: Use torch.where to avoid clone/mask assignment overhead
        # effective_cp = mask ? n_cps : next_cp
        effective_cp = torch.where(next_cp == 0, n_cps_expanded, next_cp)

        # Score calculation (Higher is better)
        # Using larger multipliers to ensure strict hierarchy: Lap > CP > Dist
        scores = (laps * 10000.0) + (effective_cp * 100.0) + dist_score
        
        # Sort scores descending to get ranks
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        
        # Creating rank tensor
        m_ranks = torch.zeros_like(scores, dtype=torch.long)
        
        # Optimized: Use pre-cached range if available, else allocate (fallback)
        if hasattr(self, 'rank_range_cache') and self.rank_range_cache.shape[0] == len(scores):
             rang = self.rank_range_cache
        else:
             rang = torch.arange(4, device=self.device).unsqueeze(0).expand(len(scores), 4)

        m_ranks.scatter_(1, sorted_indices, rang)
        
        return m_ranks

    def reset(self, env_ids=None):
        if env_ids is None:
            # All
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # 1. Generate Tracks (Vectorized Retry Logic)
        # Optimized: Fail fast (3 attempts, 20 loops) -> Fallback to Procedural Path
        
        # 1. Generate Tracks
        # Using Modular Generators for Diversity
        
        curr_env_ids = env_ids
        num_curr = len(curr_env_ids)
        
        if num_curr > 0:
            # 1. Reset CPs 
            self.checkpoints[curr_env_ids] = 0.0
            
            # 2. Random N CPs
            n_cps = torch.randint(MIN_CHECKPOINTS, MAX_CHECKPOINTS + 1, (num_curr,), device=self.device)
            self.num_checkpoints[curr_env_ids] = n_cps
            
            # 3. Select Generator Type
            if self.using_nursery_map:
                 # Override for Nursery: 3 CPs (Fixed), Simple Tracks and strictly no overlap
                 num_cps_nursery = self.config.num_checkpoints_fixed if self.config.num_checkpoints_fixed else 3
                 n_cps = torch.full((num_curr,), num_cps_nursery, device=self.device)
                 self.num_checkpoints[curr_env_ids] = n_cps
                 
                 # Generate for MAX_CHECKPOINTS but only use first n_cps
                 cps = TrackGenerator.generate_nursery_tracks(num_curr, MAX_CHECKPOINTS, WIDTH, HEIGHT, self.device)
                 self.checkpoints[curr_env_ids] = cps
                 
                 # Nursery is never predefined in the "Mad Pod Racing" sense
                 self.is_predefined_map[curr_env_ids] = False
                 
            else:
                # Standard
                # 20% Chance for Predefined Map
                # Create mask
                rand_val = torch.rand(num_curr, device=self.device)
                pred_mask = rand_val < 0.2
                
                # Split indices
                ids_pred = curr_env_ids[pred_mask]
                ids_rand = curr_env_ids[~pred_mask]
                
                # 1. Predefined
                if len(ids_pred) > 0:
                     cps_pred, num_pred = TrackGenerator.generate_predefined(len(ids_pred), MAX_CHECKPOINTS, self.device)
                     self.checkpoints[ids_pred] = cps_pred
                     self.num_checkpoints[ids_pred] = num_pred
                     self.is_predefined_map[ids_pred] = True
                     
                # 2. Random (Max Entropy)
                if len(ids_rand) > 0:
                    cps_rand = TrackGenerator.generate_max_entropy(len(ids_rand), MAX_CHECKPOINTS, WIDTH, HEIGHT, self.device)
                    self.checkpoints[ids_rand] = cps_rand
                    self.is_predefined_map[ids_rand] = False
            
        # 2. Reset Pods
        # Start at CP0
        start_pos = self.checkpoints[env_ids, 0] # [N, 2]
        
        # Look at CP1
        target_pos = self.checkpoints[env_ids, 1]
        angle_rad = torch.atan2(target_pos[:, 1] - start_pos[:, 1], target_pos[:, 0] - start_pos[:, 0])
        angle_deg = torch.rad2deg(angle_rad)
        
        # Position offsets to not overlap
        # 0: +disp, 1: -disp etc.
        # Perpendicular vector
        dx = target_pos[:, 0] - start_pos[:, 0]
        dy = target_pos[:, 1] - start_pos[:, 1]
        dist = torch.sqrt(dx**2 + dy**2) + 1e-5
        nx = -dy / dist
        ny = dx / dist
        
        # Pod 0: +400 * N
        # Pod 1: -400 * N
        # Pod 2: +1200 * N (Team 2 starts behind? Or side by side?)
        # Rules: "All pods start at the first checkpoint, facing the second."
        # Usually arranged in a grid or line.
        
        # Simple Line Arrangement
        offsets = torch.tensor(self.spawn_config.offsets, device=self.device)
        
        # Special Offsets for Predefined Maps
        # START_POINT_MULT = [[500, -500], [-500, 500], ... ]
        # Converted to tensor
        predefined_offsets = torch.tensor(START_POINT_MULT, device=self.device) # [4, 2]
        
        # Determine masks for logic
        # env_ids is the batch being reset
        # is_predefined_map[env_ids] tells us which logic to use
        is_pred_batch = self.is_predefined_map[env_ids]
        
        for i in range(4):
            # Standard Logic Position
            # pos = cp0 - offset * perp_vec
            std_pos_x = start_pos[:, 0] - nx * offsets[i] 
            std_pos_y = start_pos[:, 1] - ny * offsets[i]
            
            # Predefined Logic Position
            # pos = cp0 + explicit_offset
            # predefined_offsets[i] is [2]
            # No rotation applied to offset (as per "original mad pod racing" assumption? or aligned?)
            # Usually strict coordinate offsets.
            # "START_POINT_MULT" implies offsets from origin? No, "Start Point".
            # We assume: pos = cp0 + offset
            
            pred_pos_x = start_pos[:, 0] + predefined_offsets[i, 0]
            pred_pos_y = start_pos[:, 1] + predefined_offsets[i, 1]
            
            # Combine based on mask
            # use torch.where
            final_pos_x = torch.where(is_pred_batch, pred_pos_x, std_pos_x)
            final_pos_y = torch.where(is_pred_batch, pred_pos_y, std_pos_y)
            
            # Assign
            self.physics.pos[env_ids, i, 0] = final_pos_x
            self.physics.pos[env_ids, i, 1] = final_pos_y
            
            self.physics.vel[env_ids, i] = 0
            
            # Angle: Standard uses 'angle_deg'. 
            # Predefined? Assuming 'angle_deg' (Face next CP) is also correct for predefined maps unless specified otherwise.
            # User said "placement", likely meaning position.
            
            self.physics.angle[env_ids, i] = angle_deg
            self.physics.mass[env_ids, i] = 1.0
            self.physics.shield_cd[env_ids, i] = 0
            
            # --- Curriculum Logic (Spawn Control) ---
            # Stage 0 (Solo): Active=[0]. Others=[Infinity].
            # Stage 1 (Duel): Active=[0, 2]. Others=[Infinity].
            # Stage 2 (League): Active=[0, 1, 2, 3].
            
            active = True
            # --- Curriculum Logic (Spawn Control) ---
            active = False
            if i in self.config.active_pods:
                active = True
            
            if not active:
                # Move to infinity to avoid collision/observation noise
                self.physics.pos[env_ids, i, 0] = -100000.0
                self.physics.pos[env_ids, i, 1] = -100000.0
            
        self.physics.boost_available[env_ids] = True
        
        # Reset Game Logic
        self.next_cp_id[env_ids] = 1
        self.laps[env_ids] = 0
        
        self.timeouts[env_ids] = self.config.timeout_steps

             
        self.dones[env_ids] = False
        self.winners[env_ids] = -1
        self.cps_passed[env_ids] = 0
        self.steps_last_cp[env_ids] = 0
        
        # Update Prev Dist for rewards
        self.update_progress_metric(env_ids)
        
        # Reset Role Locks
        self.role_lock_timer[env_ids] = 0
        self._update_roles(env_ids)
        
        # Reset Ranks
        self.prev_ranks[env_ids] = self._get_ranks(env_ids)

    def set_stage(self, stage_id: int, config: EnvConfig, reset_env: bool = False):
        self.curriculum_stage = stage_id
        self.config = config
        self._update_map_mode()
        
        if reset_env:
            self.reset()

    def _update_map_mode(self):
        self.using_nursery_map = False
        if self.config.track_gen_type == "nursery":
            if self.config.mode_name == "nursery":
                self.using_nursery_map = True
            else:
                print(f"WARNING: Nursery Map requested but Mode is '{self.config.mode_name}'. Enforcing policy: FALLBACK TO MAX ENTROPY.")

    def _update_roles(self, env_ids):
        # --- FIXED ROLES OVERRIDE ---
        if self.config.fixed_roles:
            # Force roles based on config
            for pod_idx, role_id in self.config.fixed_roles.items():
                if pod_idx >= 4: continue
                # Role 1 = Runner, Role 0 = Blocker
                is_run = (role_id == 1)
                self.is_runner[env_ids, pod_idx] = is_run
            return

        if self.curriculum_stage == STAGE_DUEL_FUSED:
             # Unified Duel Stage: 50% Runner / 50% Blocker
             # Environments < NumEnvs / 2: Role 0 (Runner)
             # Environments >= NumEnvs / 2: Role 1 (Blocker)
             
             half_envs = self.num_envs // 2
             
             # Group A (First Half): Agent is Runner (True), Bot is Blocker (False)
             self.is_runner[env_ids[:half_envs], 0] = True
             self.is_runner[env_ids[:half_envs], 1] = False
             self.is_runner[env_ids[:half_envs], 2] = False # Bot is Blocker (Interceptor)
             self.is_runner[env_ids[:half_envs], 3] = False
             
             # Group B (Second Half): Agent is Blocker (False)
             # Bot (Pod 2) must be RUNNER (True) to be the target.
             # Wait, if Agent is Blocker, Bot should be Runner.
             # Config says active_pods=[0, 2].
             # Pod 0: Blocker (False)
             # Pod 2: Runner (True)
             self.is_runner[env_ids[half_envs:], 0] = False
             self.is_runner[env_ids[half_envs:], 1] = False # Teammate is Inactive (Blocker)
             self.is_runner[env_ids[half_envs:], 2] = True # Bot is Runner
             self.is_runner[env_ids[half_envs:], 3] = False
             return

        # [FIX] Force Fixed Roles for Stage 3 (Team)
        # We generally train specialized agents. Dynamic swapping ruins the Blocker's identity.
        # Stage 3 is Team -> Fixed Roles (unless explicit "Dynamic" mode requested, which is not yet supported).
        if self.curriculum_stage == STAGE_TEAM:
             # Pod 0: Runner (True)
             # Pod 1: Blocker (False)
             # Pod 2: Runner (True)
             # Pod 3: Blocker (False)
             
             self.is_runner[env_ids, 0] = True
             self.is_runner[env_ids, 1] = False
             self.is_runner[env_ids, 2] = True
             self.is_runner[env_ids, 3] = False
             return

        # Calculate Progress Score
        # Score = Laps * 1000 + NextCP * 10 + (1 - Dist/20000)
        # Higher is better
        
        # Gather data
        laps = self.laps[env_ids] # [N, 4]
        next_cp = self.next_cp_id[env_ids] # [N, 4]
        # Dist to next CP.
        # We can reuse self.prev_dist if valid, but reset might not have it yet?
        # reset calls update_progress_metric BEFORE this, so prev_dist is valid.
        dists = self.prev_dist[env_ids] # [N, 4]
        
        # Normalize dist to 0-1 (inverted, closer is better)
        # Max map dist approx 20000.
        dist_score = 1.0 - (dists / 20000.0)
        
        total_score = (laps * 1000.0) + (next_cp * 10.0) + dist_score
        
        # Compare Team 0: Pod 0 vs 1
        s0 = total_score[:, 0]
        s1 = total_score[:, 1]
        
        # Runner is argmax.
        # If s0 >= s1: Pod0 is Runner.
        # Tie-break: Pod0 (by index).
        runner0 = (s0 >= s1)
        
        # Team 1: Pod 2 vs 3
        s2 = total_score[:, 2]
        s3 = total_score[:, 3]
        runner2 = (s2 >= s3)
        
        # Hysteresis Logic
        # Filter for current envs
        current_timers = self.role_lock_timer[env_ids]
        can_update_mask = (current_timers == 0) # [len(env_ids)]
        
        # Get indices of envs that CAN update (subset of env_ids)
        # We need these to index global tensors like self.is_runner
        update_env_ids = env_ids[can_update_mask]
        
        if len(update_env_ids) > 0:
             # Extract new values for these specific envs
             new_r0 = runner0[can_update_mask]
             new_r2 = runner2[can_update_mask]
             
             # Extract old values for change detection
             old_r0 = self.is_runner[update_env_ids, 0]
             old_r2 = self.is_runner[update_env_ids, 2]
             
             # Update Global State
             self.is_runner[update_env_ids, 0] = new_r0
             self.is_runner[update_env_ids, 1] = ~new_r0
             self.is_runner[update_env_ids, 2] = new_r2
             self.is_runner[update_env_ids, 3] = ~new_r2
             
             # Change Detection
             changed0 = (old_r0 != new_r0)
             changed2 = (old_r2 != new_r2)
             
             env_changed = changed0 | changed2
             
             # Indices of envs that CHANGED
             changed_env_ids = update_env_ids[env_changed]
             
             if len(changed_env_ids) > 0:
                 self.role_lock_timer[changed_env_ids] = 50


    def update_progress_metric(self, env_ids):
        # Calculate distance to next checkpoint
        # gather next CP pos
        next_cp_indices = self.next_cp_id[env_ids] # [N, 4]
        # checkpoints: [N_total, MaxCP, 2] -> index with env_ids
        cps = self.checkpoints[env_ids] # [N, Max, 2]
        
        # Gather logic
        # We need [N, 4, 2] target positions
        # next_cp_indices is [N, 4].
        # Expand cps to gather?
        # B = len(env_ids)
        # gathered_cx = cps.gather(1, next_cp_indices)... dimension mismatch.
        # Manual lookup loop is easiest for 4 pods
        
        for i in range(4):
            cp_idx = next_cp_indices[:, i] # [N]
            # Select CP for each batch elt.
            # advanced indexing: cps[range, cp_idx]
            targets = cps[torch.arange(len(env_ids), device=self.device), cp_idx] # [N, 2]
            curr_pos = self.physics.pos[env_ids, i]
            dist = torch.norm(targets - curr_pos, dim=1)
            self.prev_dist[env_ids, i] = dist

    def step(self, actions, reward_weights, tau=0.0, team_spirit=0.0):
        """
        Stepping the Environment.
        actions: [B, 4, 4] (Thrust, Angle, Shift, Boost)
        reward_weights: [B, 13] (Per-environment weights, taken from Agent)
        tau: float, Dense Reward Annealing factor (0.0 = Full Dense, 1.0 = Full Sparse)
        team_spirit: float, 0.0=Selfish, 1.0=Cooperative. Blends rewards.
        """
        if reward_weights is None:
            # Construct default tensor
            reward_weights = torch.zeros((self.num_envs, 16), device=self.device)
            # Use default dict to fill 
            # Note: We can pre-compute this but for robust fallback:
            for k, v in DEFAULT_REWARD_WEIGHTS.items():
                reward_weights[:, k] = v
        
        # Unpack weights [B, 13]
        w_win = reward_weights[:, RW_WIN] # Sparse
        w_loss = reward_weights[:, RW_LOSS] # Sparse
        w_checkpoint = reward_weights[:, RW_CHECKPOINT] # Individual (Motor)
        w_chk_scale = reward_weights[:, RW_CHECKPOINT_SCALE] # Individual
        w_progress = reward_weights[:, RW_PROGRESS] # Individual (Replaces VELOCITY)
        w_col_run = reward_weights[:, RW_COLLISION_RUNNER] # Individual
        w_col_block = reward_weights[:, RW_COLLISION_BLOCKER] # Individual
        w_step_pen = reward_weights[:, RW_STEP_PENALTY] # Individual
        w_orient = reward_weights[:, RW_ORIENTATION] # Individual
        w_wrong_way = reward_weights[:, RW_WRONG_WAY] # Individual
        w_col_mate = reward_weights[:, RW_COLLISION_MATE] # Individual
        w_prox = reward_weights[:, RW_PROXIMITY] # Individual
        w_magnet = reward_weights[:, RW_MAGNET] # Individual (New)
        w_rank = reward_weights[:, RW_RANK] # Individual (New)
        w_lap = reward_weights[:, RW_LAP] # Lap Completion (New)
        w_denial = reward_weights[:, RW_DENIAL] # Denial Reward (New)
        
        # --- Team Spirit Blending ---
        # Modify weights based on spirit
        # Spirit 0.0 (Selfish): Base Weights.
        # Spirit 1.0 (Cooperative): Boost Win/Loss, Reduce Checkpoint/Velocity (Individual stuff).
        
        # Win/Loss Multiplier: 1.0 + team_spirit (Doubles importance at full spirit)
        w_win = w_win * (1.0 + team_spirit)
        w_loss = w_loss * (1.0 + team_spirit)
        
        # Individual Multiplier: REMOVED penalty for Team Spirit.
        # We want robust navigation always, even if cooperative.
        # indiv_mult = 1.0 - (0.5 * team_spirit) 
        indiv_mult = 1.0 # Constant
        
        w_checkpoint = w_checkpoint * indiv_mult
        w_chk_scale = w_chk_scale * indiv_mult
        w_progress = w_progress * indiv_mult
        
        # Blocker/Team Interaction:
        # Blocker Reward IS a team contribution, so maybe boost it?
        # Collision Mate Penalty: Boost it (Don't hit friends!)
        w_col_mate = w_col_mate * (1.0 + team_spirit)
        
        # Unpack Actions & Clamp
        # Thrust: [0..1] -> [0..100]
        act_thrust = torch.clamp(actions[..., 0], 0.0, 1.0) * 100.0
        # Angle: [-1..1] -> [-18..18]
        act_angle = torch.clamp(actions[..., 1], -1.0, 1.0) * 18.0
        # Shield: > 0.5
        act_shield = actions[..., 2] > 0.5
        # Boost: > 0.5
        act_boost = actions[..., 3] > 0.5
        
        # --- Bot Logic (Same as before) ---
        if self.config.use_bots:
            bot_pods = self.config.bot_pods
            
            for bot_id in bot_pods:
                # Override Actions with Simple Bot
                
                # Check Role
                is_bot_runner = self.is_runner[:, bot_id] # [B]
                
                # --- VECTORIZED TARGET LOGIC ---
                # We must compute both Targets (Runner/Blocker) and blend, 
                # because branching 'if is_bot_runner' fails on mixed batches.

                batch_indices = torch.arange(self.num_envs, device=self.device)
                
                # A. RUNNER TARGET (Checkpoint)
                opp_nid = self.next_cp_id[:, bot_id]
                runner_target = self.checkpoints[batch_indices, opp_nid]

                # B. BLOCKER TARGET (Intercept Agent)
                agent_pos = self.physics.pos[:, 0]
                bot_pos = self.physics.pos[:, bot_id]
                
                # Agent's Target CP
                agent_next_cp = self.next_cp_id[:, 0]
                cp_pos = self.checkpoints[batch_indices, agent_next_cp]
                
                vec_runner_to_cp = cp_pos - agent_pos
                dist_runner_to_cp = torch.norm(vec_runner_to_cp, dim=1)
                div_dist = dist_runner_to_cp.unsqueeze(1) + 1e-6
                dir_runner_to_cp = vec_runner_to_cp / div_dist

                # Project Bot Position? No, project "Intercept Point"
                # "Gatekeeper": 1500u in front of CP
                # [FIX]: Scale Offset with Difficulty (High Diff = Tighter Guard = Smaller Offset)
                # Diff 0 -> 2500, Diff 1 -> 500
                diff_factor = self.bot_difficulty
                offset_val = 2500.0 - (2000.0 * diff_factor) # Linear from 2500 to 500
                intercept_pos = cp_pos - (dir_runner_to_cp * offset_val)

                # Emergency Ram: If Agent is close to CP (<2000), target Agent directly
                close_mask = (dist_runner_to_cp < 2000.0).unsqueeze(1) # [B, 1]
                blocker_target = torch.where(close_mask, agent_pos, intercept_pos)

                # C. SELECT TARGET based on Role
                # is_bot_runner: [B] -> [B, 1]
                role_mask = is_bot_runner.unsqueeze(1)
                target_pos = torch.where(role_mask, runner_target, blocker_target)

                # D. STEERING CALCULATION
                diff = target_pos - bot_pos
                # variable 'dist_to_cp' is used for thrust scaling later. 
                # Let's call it 'dist_to_target' to be generic.
                dist_to_target = torch.norm(diff, dim=1)
                
                desired_rad = torch.atan2(diff[:, 1], diff[:, 0])
                desired_deg = torch.rad2deg(desired_rad)

                # --- Dynamic Difficulty Scaling ---
                
                # --- Dynamic Difficulty Scaling ---
                # 1. Steering Noise
                curr_difficulty = self.bot_difficulty
                noise_scale = (1.0 - curr_difficulty) * self.bot_config.difficulty_noise_scale
                
                noise = (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * noise_scale
                desired_deg += noise
                
                # Current Angle
                curr_deg = self.physics.angle[:, bot_id]
                
                # Delta
                delta = desired_deg - curr_deg
                # Normalize -180..180
                delta = (delta + 180) % 360 - 180
                
                # Clamp to -18..18
                delta = torch.clamp(delta, -18.0, 18.0)
                
                # Set Actions
                act_angle[:, bot_id] = delta
                
                # 2. Thrust Scaling
                # 20 + (80 * diff)
                thrust_val = self.bot_config.thrust_base + (self.bot_config.thrust_scale * curr_difficulty)
                
                # [FIX] Ramming Speed: If aiming at target (Runner) and close, BOOST thrust
                # blocker_target vs runner_target?
                # We want to know if we are in "Ramming Mode".
                # If we are blocker, and target is runner (or intercept point close to runner), and aligned.
                if not is_bot_runner[bot_id]:
                     # Check alignment
                     # diff already calc.
                     # dist_to_target already calc.
                     # If aligned within 20 deg and dist < 4000
                     # Add extra thrust
                     is_aligned = torch.abs(delta) < 20.0
                     is_close_enough = dist_to_target < 4000.0
                     
                     ram_mask = is_aligned & is_close_enough
                     # Add 20-30 thrust
                     thrust_val = torch.where(ram_mask, thrust_val + self.bot_config.ramming_speed_scale, thrust_val)
                
                # [FIX] Bot Stabilization: Slow down near checkpoints to avoid "Orbiting"
                # If too fast, the bot cannot turn sharply enough to hit the radius.
                # [FIXED] Bot Stabilization: Less aggressive slow down
                # Using generic 'dist_to_target' 
                slow_down_mask = (dist_to_target < 1500.0)
                thrust_val = torch.where(slow_down_mask, thrust_val * 0.8, thrust_val)
                very_close_mask = (dist_to_target < 600.0)
                thrust_val = torch.where(very_close_mask, thrust_val * 0.5, thrust_val)
                
                act_thrust[:, bot_id] = thrust_val 
                
                act_shield[:, bot_id] = False
                act_boost[:, bot_id] = False
        
        # Track Previous Position for Continuous Collision Detection (CCD)
        prev_pos = self.physics.pos.clone()

        # 1. Physics Step
        collisions, collision_vectors = self.physics.step(act_thrust, act_angle, act_shield, act_boost)
        
        # Calculate Collision Flags for Telemetry [Batch, 4]
        col_sums = collisions.sum(dim=2)
        collision_flags = (col_sums > 0).float()
        
        # 2. Reward Containers
        rewards_indiv = torch.zeros((self.num_envs, 4), device=self.device) # Pure Individual
        rewards_team = torch.zeros((self.num_envs, 4), device=self.device) # Pure Team (Win/Loss)

        # Metric Helpers
        infos = {
             "laps_completed": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device),
             "checkpoints_passed": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device),
             "current_streak": self.cps_passed.clone(),
             "cp_steps": torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device), # Steps taken for passed CPs
             "collision_flags": collision_flags # [Batch, 4]
        }

        # --- DENSE REWARDS (Individual) ---
        new_dists = torch.zeros_like(self.prev_dist)
        
        # Annealing
        dense_mult = (1.0 - tau)
        if isinstance(dense_mult, torch.Tensor):
             dense_mult = dense_mult.squeeze()
        
        for i in range(4):
            next_ids = self.next_cp_id[:, i]
            batch_indices = torch.arange(self.num_envs, device=self.device)
            target_pos = self.checkpoints[batch_indices, next_ids]
            curr_pos = self.physics.pos[:, i]
            
            # --- A. Progress Reward (Prev Dist - Curr Dist) ---
            new_dists[:, i] = torch.norm(target_pos - curr_pos, dim=1)
            
            # Positive = Approach, Negative = Retreat
            progress = self.prev_dist[:, i] - new_dists[:, i]
            
            # Apply Progress Reward
            # Note: 1 unit distance = 1 point if weight=1.0. 
            # Apply Progress Reward
            # Note: 1 unit distance = 1 point if weight=1.0. 
            # [FIX] Mask for Blockers in Stage 3+
            if self.curriculum_stage >= STAGE_DUEL_FUSED:
                is_run = self.is_runner[:, i].float()
                rewards_indiv[:, i] += progress * w_progress * dense_mult * is_run
            else:
                rewards_indiv[:, i] += progress * w_progress * dense_mult
            
            # --- B. Magnet Reward (Proximity Center) ---
            # If inside Approach Radius (e.g. 2 * Checkpoint Radius or just Checkpoint Radius?)
            # Let's say we pull them in from 1.5x Radius.
            MAGNET_RADIUS = CHECKPOINT_RADIUS * 1.5
            dist = new_dists[:, i]
            
            in_magnet_mask = dist < MAGNET_RADIUS
            if in_magnet_mask.any():
                # Score 0.0 to 1.0 (Close)
                magnet_score = (1.0 - (dist[in_magnet_mask] / MAGNET_RADIUS))
                # Square it to make the center much more attractive than the edge? No, linear is stable.
                # Score 0.0 to 1.0 (Close)
                magnet_score = (1.0 - (dist[in_magnet_mask] / MAGNET_RADIUS))
                # Square it to make the center much more attractive than the edge? No, linear is stable.
                
                # [FIX] Mask for Blockers
                if self.curriculum_stage >= STAGE_DUEL_FUSED:
                     is_run = self.is_runner[in_magnet_mask, i].float()
                     rewards_indiv[in_magnet_mask, i] += magnet_score * w_magnet[in_magnet_mask] * dense_mult * is_run
                else:
                     rewards_indiv[in_magnet_mask, i] += magnet_score * w_magnet[in_magnet_mask] * dense_mult
        
        # --- C. Orientation Reward (Soft Guidance / Wrong Way) ---
        if w_orient.sum() > 0.0 or w_wrong_way.sum() > 0.0:
            for i in range(4):
                next_ids = self.next_cp_id[:, i]
                batch_indices = torch.arange(self.num_envs, device=self.device)
                target_pos = self.checkpoints[batch_indices, next_ids] 
                curr_pos = self.physics.pos[:, i] 
                
                diff = target_pos - curr_pos
                target_angle = torch.atan2(diff[:, 1], diff[:, 0])
                curr_angle = torch.deg2rad(self.physics.angle[:, i])
                alignment = torch.cos(target_angle - curr_angle)

                # Positive Reward (Soft Guidance)
                if w_orient.sum() > 0.0:
                    THRESHOLD = self.reward_scaling_config.orientation_threshold
                    pos_score = torch.clamp((alignment - THRESHOLD) / (1.0 - THRESHOLD), 0.0, 1.0)
                    THRESHOLD = self.reward_scaling_config.orientation_threshold
                    pos_score = torch.clamp((alignment - THRESHOLD) / (1.0 - THRESHOLD), 0.0, 1.0)
                    
                    # [FIX] Mask for Blockers
                    if self.curriculum_stage >= STAGE_DUEL_FUSED:
                         is_run = self.is_runner[:, i].float()
                         rewards_indiv[:, i] += pos_score * w_orient * dense_mult * is_run
                    else:
                         rewards_indiv[:, i] += pos_score * w_orient * dense_mult

                # Negative Penalty (Wrong Way)
                neg_mask = alignment < -0.5 # Strictly facing away
                if neg_mask.any():
                    # Heavy penalty for wrong way
                    rewards_indiv[neg_mask, i] += alignment[neg_mask] * w_wrong_way[neg_mask]

        # --- D. Rank Reward (Potential-Based) ---
        # "Overtaking Reward"
        w_rank = reward_weights[:, RW_RANK]
        if w_rank.sum() > 0.0:
            # Calculate current ranks
            # Optimized: Pass slice(None) to avoid arange(num_envs) allocation
            curr_ranks = self._get_ranks(slice(None))
            
            # Potential diff: Prev - Curr
            # Rank 1 (2nd) -> Rank 0 (1st). Diff = 1 - 0 = +1 (Improvement)
            # Rank 0 (1st) -> Rank 1 (2nd). Diff = 0 - 1 = -1 (Loss)
            rank_diff = self.prev_ranks - curr_ranks
            
            # Apply Reward for each pod
            # Note: rank_diff is [N, 4]
            # w_rank is [N]. Expand.
            # FIX: Only apply to Runners. Blockers should not be penalized for losing rank (ambush).
            is_runner_mask = self.is_runner.float() # [B, 4]
            # Apply Reward for each pod
            # Note: rank_diff is [N, 4]
            # w_rank is [N]. Expand.
            # FIX: Only apply to Runners. Blockers should not be penalized for losing rank (ambush).
            # [FIX] Already had is_runner_mask logic planned, ensuring it is used.
            is_runner_mask = self.is_runner.float() # [B, 4]
            # If Stage < 3, everyone is runner essentially (or we don't care). 
            # But is_runner is managed correctly in all stages.
            rewards_indiv += rank_diff.float() * w_rank.unsqueeze(1) * is_runner_mask
            
            # Update state
            self.prev_ranks = curr_ranks


        # --- D. Step Penalty (Constant) ---
        # Fixed negative reward per step to encourage speed.
        if w_step_pen.sum() > 0:
            active_mask = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
            for i in self.config.step_penalty_active_pods:
                active_mask[:, i] = True
                
            # Apply to active pods
            # w_step_pen is positive in config (e.g. 1.0), so we subtract it.
            # Assuming w_step_pen broadcastable [B]
            pen_val = w_step_pen.unsqueeze(1).expand(-1, 4)
            
            # Only apply where mask is true
            # rewards_indiv -= pen_val * active_mask.float()
            # Careful with shape
            penalty_tensor = pen_val * active_mask.float()
            
            # FIX: Mask penalty for Blockers to allow "Doorman" loitering
            # Blockers (is_runner=0) get 0.0 * penalty
            # Runners (is_runner=1) get 1.0 * penalty
            role_scale = self.is_runner.float() # [B, 4] -> 1.0 for Runner, 0.0 for Blocker
            
            # Apply Penalty
            rewards_indiv -= penalty_tensor * role_scale

        
        # D. Checkpoints (Individual Progress)
        # For each pod, check distance to next_cp
        # Logic: dist < 600 -> passed.
        
        all_cp_pos = self.checkpoints # [B, Max, 2]
        
        events_cp_passed = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device) # For visuals/logging
        
        # --- Rank Calculation (Global) ---
        # Score = Lap * 50000 + NextCP * 500 + (20000 - Dist)
        p_laps = self.laps.float()
        p_ncp = self.next_cp_id.float()
        # new_dists is [B, 4] calculated in Dense Rewards section
        true_dist = new_dists 
        
        scores = p_laps * 50000.0 + p_ncp * 500.0 + (20000.0 - true_dist)
        
        # Calculate Rank
        s_col = scores.unsqueeze(2)
        s_row = scores.unsqueeze(1)
        ranks = (s_row > s_col).sum(dim=2).float().unsqueeze(-1) # [B, 4, 1]
        rank_norm = ranks / 3.0 # [0, 1]
        
        for i in range(4):
            next_ids = self.next_cp_id[:, i]
            # Gather targets
            # use unsqueeze exp.
            batch_indices = torch.arange(self.num_envs, device=self.device)
            target_pos = all_cp_pos[batch_indices, next_ids] # [B, 2]
            
            p_pos = self.physics.pos[:, i] # [B, 2]
            
            # --- Continuous Collision Detection (Segment-Circle Intersection) ---
            # Segment: prev_pos[:, i] -> p_pos
            # Circle: target_pos, Radius 600
            
            p_prev = prev_pos[:, i]
            
            # Vector A->B
            seg_v = p_pos - p_prev
            seg_len_sq = (seg_v ** 2).sum(dim=1)
            
            # Vector A->C (Center)
            ac_v = target_pos - p_prev
            
            # Project C onto Line AB: t = (AC . AB) / |AB|^2
            t = (ac_v * seg_v).sum(dim=1) / (seg_len_sq + 1e-6)
            
            # Clamp t to segment [0, 1]
            t = torch.clamp(t, 0.0, 1.0)
            
            # Closest Point
            closest = p_prev + seg_v * t.unsqueeze(1)
            
            # Dist Closest -> Center
            dist_sq = ((closest - target_pos) ** 2).sum(dim=1)
            
            # Check pass
            # Radius 600
            passed = dist_sq < (CHECKPOINT_RADIUS ** 2)
            
            # Update State
            if passed.any():
                pass_idx = torch.nonzero(passed).squeeze(-1)
                
                # Reset timeout
                # Reset timeout
                self.timeouts[pass_idx, i] = self.config.timeout_steps 
                
                # [FIX] Team Resets: In Stage 4 (Team), Runner resets Blocker's timer too.
                # Just reset the whole team? Or specifically the teammate?
                # Teammate index = i ^ 1
                if self.curriculum_stage >= STAGE_TEAM:
                     mate_idx = i ^ 1
                     self.timeouts[pass_idx, mate_idx] = self.config.timeout_steps 
                
                # Determine Next CP and Lap Logic
                # Use cached 'next_ids' (OLD target)
                old_target_id = next_ids[passed]
                
                # Standard increment
                new_target_id = (old_target_id + 1) 
                
                # Wrap
                limits = self.num_checkpoints[pass_idx] # [P]
                wrapped = new_target_id % limits
                
                # Apply Update
                self.next_cp_id[pass_idx, i] = wrapped
                
                # Check Lap Completion
                # If we passed CP 0 (Start/Finish Line), we completed a lap.
                # CP0 is ID 0.
                just_passed_zero = (old_target_id == 0)
                
                if just_passed_zero.any():
                    z_idx = pass_idx[just_passed_zero]
                    self.laps[z_idx, i] += 1
                    infos["laps_completed"][z_idx, i] = 1
                    
                    # --- Curriculum Metric: Solo Complete ---
                    if self.curriculum_stage == STAGE_NURSERY or self.curriculum_stage == STAGE_SOLO:
                        # If Pod 0 finished a lap
                        if i == 0:
                             self.stage_metrics["solo_completes"] += len(z_idx)
                

                
                # --- Metric: Checkpoints Passed ---
                # Only count for active pod (Pod 0 in Solo)?
                # Let's count all active pods to be fair metric of "system activity".
                # But user wants "progress".
                if self.curriculum_stage == STAGE_NURSERY or self.curriculum_stage == STAGE_SOLO:
                    if i == 0:
                         self.stage_metrics["checkpoint_hits"] += len(pass_idx)
                else:
                    self.stage_metrics["checkpoint_hits"] += len(pass_idx)

                # w.get("checkpoint", 500.0) -> Add to buffer
                # Progressive Reward: Base + (Streak * Scale)
                # Base is now 500.0 as requested.
                # Assuming increment is also 500.0?
                # Or use the weight as base?
                # original plan: base=4000. user: make base 500.
                # Let's say w["checkpoint"] is the BASE.
                # And we add a scaling factor.
                
                # w.get("checkpoint", 2000.0) -> Base

                # Progressive Reward: Base + (Streak * Scale) + DYNAMIC DECAY
                
                self.cps_passed[pass_idx, i] += 1
                infos["checkpoints_passed"][pass_idx, i] = 1
                streak = self.cps_passed[pass_idx, i].float() # 1, 2, 3...
                
                # Record Efficiency (Steps taken)
                taken_steps = self.steps_last_cp[pass_idx, i]
                infos["cp_steps"][pass_idx, i] = taken_steps
                self.steps_last_cp[pass_idx, i] = 0
                
                
                # --- REWARD CALCULATION ---
                # RW_LAP imported from config
                
                w_cp_base = w_checkpoint[pass_idx]
                w_lap_base = reward_weights[pass_idx, RW_LAP] if reward_weights.shape[1] > RW_LAP else torch.full((len(pass_idx),), 2000.0, device=self.device)
                
                # Split Logic: Lap vs Normal CP
                # 'just_passed_zero' is boolean mask for pass_idx
                
                # 1. Normal CP Rewards
                # Apply to ~just_passed_zero
                normal_mask = ~just_passed_zero
                
                # Calculate Normal Reward for ALL (we will mask later)
                streak_bonus = (streak - 1.0) * w_chk_scale[pass_idx]
                reward_normal = w_cp_base + streak_bonus
                
                # 2. Lap Rewards
                current_laps = self.laps[pass_idx, i]
                lap_mult_pow = torch.clamp(current_laps - 1, min=0).float()
                
                # Multiplier
                mult = torch.pow(LAP_REWARD_MULTIPLIER, lap_mult_pow)
                reward_lap = w_lap_base * mult
                
                # Combine
                # If just_passed_zero: use reward_lap
                # Else: use reward_normal
                total_reward = torch.where(just_passed_zero, reward_lap, reward_normal)

                # --- TIME EXTENSION ---
                # Reset Timeout AFTER reward calc
                self.timeouts[pass_idx, i] = self.config.timeout_steps
                
                # [FIX] Team Resets: Stage 4+
                if self.curriculum_stage >= STAGE_TEAM:
                     mate_idx = i ^ 1
                     self.timeouts[pass_idx, mate_idx] = self.config.timeout_steps 
                
                # DIRECT REWARD (Individual)
                # [FIX] Role Preservation: Blockers get ZERO Racing Rewards in Stage 3+
                if self.curriculum_stage >= STAGE_DUEL_FUSED: # Stage 3
                     is_run_mask = self.is_runner[pass_idx, i].float()
                     total_reward = total_reward * is_run_mask
                
                rewards_indiv[pass_idx, i] += total_reward

                # --- NEW [Proposal B]: Goalie's Burden (Opponent CP Penalty) ---
                # If opponent crosses, punish the blocker.
                if self.curriculum_stage >= STAGE_DUEL_FUSED:
                    # Identify Opponent Blockers
                    # i is the pod that passed.
                    # team = i // 2. Opponent team = 1 - team.
                    team = i // 2
                    opp_team = 1 - team
                    
                    # Opponent indices
                    # e.g. if team=0, opp=1 -> pods 2,3
                    o1 = 2 * opp_team
                    o2 = 2 * opp_team + 1
                    
                    # Penalty Amount (Hardcoded Sharp Penalty)
                    # -1000.0 (Equivalent to giving up 2 standard CPs)
                    PENALTY_VAL = -1000.0 
                    
                    # Apply to Opponent Blockers ONLY
                    for opp_idx in [o1, o2]:
                        is_blocker = ~self.is_runner[pass_idx, opp_idx] # Boolean mask [K]
                        
                        if is_blocker.any():
                            # Apply penalty
                            rewards_indiv[pass_idx, opp_idx] += (PENALTY_VAL * is_blocker.float())
                
                # Correction for Dense Reward Overshoot
                overshoot_dist = new_dists[pass_idx, i]
                
                # [FIX] Role Preservation: Mask Progress too
                if self.curriculum_stage >= STAGE_DUEL_FUSED:
                     is_run_mask = self.is_runner[pass_idx, i].float()
                     rewards_indiv[pass_idx, i] += overshoot_dist * w_progress[pass_idx] * dense_mult * is_run_mask
                else:
                     rewards_indiv[pass_idx, i] += overshoot_dist * w_progress[pass_idx] * dense_mult

        # Calculate dist_to_next for Nursery Metric [B, 4]
        # We need distance to next checkpoint for ALL pods, regardless of passing.
        # This is essentially 'new_dists' calculated earlier at line 447.
        infos['dist_to_next'] = new_dists
        
        # Increment Step Counter for Efficiency
        self.steps_last_cp += 1

        # 3. Timeouts (Loss Condition)
        self.timeouts -= 1
        timed_out = (self.timeouts <= 0)
        env_timed_out = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        if self.curriculum_stage == STAGE_SOLO:
            env_timed_out = timed_out[:, 0]
        elif self.curriculum_stage == STAGE_DUEL_FUSED:
            # Fused Stage: Reset if ANY Runner times out.
            runner_timed_out = (timed_out & self.is_runner).any(dim=1)
            env_timed_out = runner_timed_out
            
            # Handling Denial Rewards & Winner Codes
            if env_timed_out.any():
                t_idx = torch.nonzero(env_timed_out).squeeze(-1)
                
                # Check Role of Agent (Pod 0)
                # If Pod 0 is Blocker, it successfully Denied the Bot Runner -> Award Denial
                # If Pod 0 is Runner, it failed to finish (Bot Blocker won) -> No Denial Reward
                
                p0_is_blocker = ~self.is_runner[t_idx, 0]
                
                # 1. Award Explicit Denial Reward to Agent Blocker
                # (Only if Agent is Blocker)
                rewards_indiv[t_idx, 0] += p0_is_blocker.float() * w_denial[t_idx]
                
                # 2. Set Winner Code
                # -1 = Agent Denial (Agent was Blocker and stopped Runner)
                #  1 = Bot Win (Agent was Runner and timed out)
                
                # Init with 1 (Bot Win)
                self.winners[t_idx] = 1 
                # Override with -1 where Agent is Blocker
                if p0_is_blocker.any():
                    # We need to index into t_idx, then into winners
                    denial_indices = t_idx[p0_is_blocker]
                    self.winners[denial_indices] = -1
                    # [METRIC] Count Denials
                    if self.curriculum_stage >= STAGE_DUEL_FUSED:
                         self.stage_metrics["recent_denials"] += len(denial_indices)
            
            # [FIX]: Count finished games (Runner Finishes) for Denial Rate
            # If Runner wins (reaches lap limit), it's a finish.
            # Logic below handles 'env_won' which triggers 'dones'.
            # We need to hook into 'env_won' logic.
        elif self.curriculum_stage == STAGE_TEAM:
            # Stage 4: Team 2v2.
            # Strategic Timeout Activated: Any pod (Agent or Bot) triggers reset.
            # This allows "Stalling an opponent" to be a win condition.
            env_timed_out = timed_out.any(dim=1)
        else:
            # League or others: Any timeout resets
            env_timed_out = timed_out.any(dim=1)

        self.dones = self.dones | env_timed_out
        
        # --- Degressive Timeout Penalty ---
        # Penalize agents for failing to finish, scaled by how much they failed.
        # Concept: "Billing for Wasted Time"
        
        if env_timed_out.any() and self.curriculum_stage != STAGE_NURSERY:
            # Identify indices
            idx = torch.nonzero(env_timed_out).squeeze(-1)
            
            # Apply to all 4 pods (if they timed out logic applies)
            # In Solo/Duel, 'env_timed_out' is OR of active pods.
            # We strictly penalize the specific pod that caused timeout? 
            # Or assume global timeout applies to all?
            # 'timed_out' is [B, 4]. Let's use that for precision.
            
            for p_i in range(4):
                pod_timeouts = timed_out[:, p_i]
                if pod_timeouts.any():
                     p_idx = torch.nonzero(pod_timeouts).squeeze(-1)
                     
                     # Calculate Progress
                     passed = self.cps_passed[p_idx, p_i].float()
                     
                     # Total Goal (Checkpoints * Laps)
                     # self.num_checkpoints is [B]
                     total_goal = (self.num_checkpoints[p_idx] * MAX_LAPS).float()
                     
                     progress = torch.clamp(passed / total_goal, 0.0, 1.0)
                     
                     # Fixed Rate 25.0 * 100 Steps = 2500.0
                     # [CRITICAL UPDATE] Timeout Penalty must exceed Loss Penalty (5000.0)
                     # otherwise agents prefer to spin/stall than race and lose.
                     if self.curriculum_stage == STAGE_DUEL_FUSED:
                         MAX_PENALTY = TIMEOUT_PENALTY_DUEL # Huge penalty to force finishing
                     else:
                         MAX_PENALTY = TIMEOUT_PENALTY_STANDARD
                     
                     penalty = MAX_PENALTY * (1.0 - progress)
                     
                     if self.curriculum_stage >= STAGE_DUEL_FUSED:
                          is_run = self.is_runner[p_idx, p_i].float() # [K]
                          rewards_indiv[p_idx, p_i] -= penalty * is_run
                     else:
                          rewards_indiv[p_idx, p_i] -= penalty
        finished = (self.laps >= MAX_LAPS)
        
        if finished.any():
            env_won = finished.any(dim=1)
            
            winner_indices = torch.nonzero(env_won).squeeze(-1)
            
            t0_wins = (finished[:, 0] | finished[:, 1])
            t1_wins = (finished[:, 2] | finished[:, 3])
            
            self.winners[t0_wins] = 0
            self.winners[t1_wins & ~t0_wins] = 1
            
            self.dones[env_won] = True
            
            # --- SPARSE REWARDS (Team Shared) ---
            # 1. Win/Loss is inherently Team-based.
            # 2. We apply to 'rewards_team'.
            
            mask_w0 = (self.winners == 0) & env_won
            mask_w1 = (self.winners == 1) & env_won
            
            # Team 0 Rewards
            # Team 0 Rewards
            # [FIX] Disable Win Reward in Stage 3 (Blocker Academy)
            if self.curriculum_stage != STAGE_DUEL_FUSED:
                rewards_team[mask_w0, 0] += w_win[mask_w0]
                rewards_team[mask_w0, 1] += w_win[mask_w0]
            
            rewards_team[mask_w0, 2] -= w_loss[mask_w0]
            rewards_team[mask_w0, 3] -= w_loss[mask_w0]
            
            # Team 1 Rewards
            # Team 1 Rewards
            # Always give rewards to Team 1 (Bot/Enemy)
            rewards_team[mask_w1, 2] += w_win[mask_w1]
            rewards_team[mask_w1, 3] += w_win[mask_w1]
            
            rewards_team[mask_w1, 0] -= w_loss[mask_w1]
            rewards_team[mask_w1, 1] -= w_loss[mask_w1]
            
            # Metrics
            # Metrics
            if self.curriculum_stage == STAGE_DUEL_FUSED:
                n_wins = mask_w0.sum().item()
                n_games = env_won.sum().item()
                self.stage_metrics["duel_wins"] += n_wins
                self.stage_metrics["duel_games"] += n_games
                self.stage_metrics["recent_wins"] += n_wins
                self.stage_metrics["recent_games"] += n_games
                
            elif self.curriculum_stage == STAGE_TEAM:
                n_wins = mask_w0.sum().item() 
                n_games = env_won.sum().item() 
                self.stage_metrics["recent_wins"] += n_wins
                self.stage_metrics["recent_games"] += n_games
        
        S_VEL = 1.0 / 1000.0
        self.update_progress_metric(torch.arange(self.num_envs, device=self.device))
        
        num_dones = self.dones.sum().item()
        if num_dones > 0:
             self.stage_metrics["recent_episodes"] += num_dones
        
        # --- Role Collision Rewards (Individual) ---
        runner_velocity_metric = torch.zeros((self.num_envs, 4), device=self.device)
        blocker_damage_metric = torch.zeros((self.num_envs, 4), device=self.device)
        
        for i in range(4):
            team = i // 2
            enemy_team = 1 - team
            enemy_indices = [2*enemy_team, 2*enemy_team + 1]
            
            is_run = self.is_runner[:, i] # [B]
            
            v_mag = torch.norm(self.physics.vel[:, i], dim=1)
            runner_velocity_metric[:, i] = v_mag * is_run.float() * S_VEL 
            
            impact_e1 = collisions[:, i, enemy_indices[0]]
            impact_e2 = collisions[:, i, enemy_indices[1]]
            total_impact = impact_e1 + impact_e2
            
            # 1. Runner Penalty
            runner_pen = -w_col_run * total_impact
            rewards_indiv[:, i] += runner_pen * is_run.float()
            
            # [REFINE] Blocker Collision Reward (Energy Transfer + Direction)
            is_block = ~is_run
            
            bonus_blocker = torch.zeros(self.num_envs, device=self.device)
            total_impact_force = torch.zeros(self.num_envs, device=self.device)
            
            # Iterate Enemies
            for e_idx in [0, 1]:
                 real_e_idx = enemy_indices[e_idx] # Tensor scalar? No list.
                 # Use tensor indexing if needed, but loop is over 2 static indices.
                 idx_e = enemy_indices[e_idx] # int
                 
                 # Impulse ON Enemy FROM Me (Me=i, Enemy=idx_e)
                 # vectors uses [B, target, source] or [B, source, target]?
                 # gpu_physics: impacts[:, i1, i2] += val. impact_vectors[:, i1, i2] += vec_j (On p1 from p2?)
                 # No, vec_j was J on P1.
                 # So impact_vectors[B, P1, P2] is Impulse on P1 from P2.
                 # We want Impulse on Enemy (P1=idx_e) from Me (P2=i).
                 # J_on_e = vectors[:, idx_e, i]
                 
                 j_vec = collision_vectors[:, idx_e, i] # [B, 2]
                 j_mag = torch.norm(j_vec, dim=1)
                 
                 total_impact_force += j_mag
                 
                 # Direction Check
                 # Vector Enemy -> NextCP
                 e_pos = self.physics.pos[:, idx_e]
                 # We need NextCP of Enemy
                 e_next_cp = self.next_cp_id[:, idx_e]
                 e_cp_pos = self.checkpoints[torch.arange(self.num_envs), e_next_cp]
                 
                 e_dir_to_cp = e_cp_pos - e_pos
                 e_dist_to_cp = torch.norm(e_dir_to_cp, dim=1) + 1e-6
                 e_dir_norm = e_dir_to_cp / e_dist_to_cp.unsqueeze(1)
                 
                 # Dot Product (J . Dir)
                 # Positive = Pushing Towards (Helping) -> Penalize or Zero
                 # Negative = Pushing Away (Hurting) -> Reward
                 dot = (j_vec * e_dir_norm).sum(dim=1)
                 
                 # Reward pushing AWAY (dot < 0)
                 # helpful_push = -dot (positive when dot is negative)
                 helpful_push = torch.clamp(-dot, min=0.0)
                 
                 # Scale by Energy? helpful_push is Energy projected.
                 # So just helpful_push is the magnitude of the "Effective Impulse".
                 
                 # Accumulate Bonus
                 # [FIX] Rank Scaling? Hitting leader is better.
                 r_e = rank_norm[:, idx_e, 0]
                 s_e = 1.0 + (1.0 - r_e)
                 
                 # Apply
                 bonus_blocker += helpful_push * s_e * self.is_runner[:, idx_e].float() # Only reward hitting runners
                 
            # Apply Reward
            rewards_indiv[:, i] += bonus_blocker * w_col_block * dense_mult * is_block.float()
            
            # [METRIC] Accumulate Impact (Raw Magnitude)
            # Only count if I am a blocker
            is_block_f = is_block.float()
            self.stage_metrics["blocker_impact"] += (total_impact_force * is_block_f).sum().item()
            
            # --- RW_ZONE (Intercept Control) ---
            # w_zone weight index 16
            w_zone = reward_weights[:, RW_ZONE] if reward_weights.shape[1] > RW_ZONE else torch.zeros(self.num_envs, device=self.device)
            
            if w_zone.sum() > 0 and self.curriculum_stage >= STAGE_DUEL_FUSED:
                 # Calculate Intercept Point (Simplified)
                 # Target: Closest/Leading Enemy Runner
                 # We assume pod 'i' is blocker.
                 
                 # Identify target runner (best rank among enemies)
                 # This logic is complex to vectorize for 4 pods.
                 # For now, calc for all blockers against *their* target.
                 # Bot logic calculated "dir_runner_to_cp".
                 # We replicate:
                 
                 # Find Primary Target (Best Rank)
                 # enemies: enemy_indices
                 r0 = rank_norm[:, enemy_indices[0], 0]
                 r1 = rank_norm[:, enemy_indices[1], 0]
                 # Target index 0 or 1 of enemy pair?
                 # If r0 < r1 (Rank 0 is better/lower), target 0.
                 target_local_idx = (r1 < r0).long() # 0 or 1
                 target_idx = torch.gather(torch.tensor(enemy_indices, device=self.device).unsqueeze(0).expand(self.num_envs, 2), 1, target_local_idx.unsqueeze(1)).squeeze(1)
                 
                 # Target state
                 t_pos = self.physics.pos[torch.arange(self.num_envs), target_idx]
                 t_cp_id = self.next_cp_id[torch.arange(self.num_envs), target_idx]
                 t_cp_pos = self.checkpoints[torch.arange(self.num_envs), t_cp_id]
                 
                 t_dir = t_cp_pos - t_pos
                 t_dir = t_dir / (torch.norm(t_dir, dim=1, keepdim=True) + 1e-6)
                 
                 # Intercept Point: 1500u in front of CP (Generic Gatekeeper)
                 # Or use Bot Logic scaling?
                 # Let's reward being at the "Gatekeeper" spot: 1500u from CP.
                 intercept_pt = t_cp_pos - (t_dir * 1500.0)
                 
                 # My Pos
                 my_pos = self.physics.pos[:, i]
                 dist_to_spot = torch.norm(intercept_pt - my_pos, dim=1)
                 
                 # Reward: exp(-dist / 1000)
                 zone_rew = torch.exp(-dist_to_spot / 1000.0)
                 
                 rewards_indiv[:, i] += zone_rew * w_zone * dense_mult * is_block.float()
        
            # 3. Teammate Collision Penalty
            mate_idx = i ^ 1
            impact_mate = collisions[:, i, mate_idx]
            mate_pen = -w_col_mate * impact_mate
            rewards_indiv[:, i] += mate_pen
            
        # --- Proximity Reward (Blocker -> Enemy Runner) ---
        if self.curriculum_stage >= STAGE_DUEL_FUSED:
             for i in range(4):
                 is_block = ~self.is_runner[:, i]
                 if not is_block.any(): continue
                 
                 team = i // 2
                 enemy_team = 1 - team
                 enemy_indices = torch.tensor([2*enemy_team, 2*enemy_team + 1], device=self.device)
                 
                 e_runner_mask = self.is_runner[:, enemy_indices]
                 prox_rew = torch.zeros(self.num_envs, device=self.device)
                 p_pos = self.physics.pos[:, i]
                 
                 for e_idx in [0, 1]:
                      real_e_idx = enemy_indices[e_idx] # tensor
                      real_e = 2 * enemy_team + e_idx
                      e_pos = self.physics.pos[:, real_e]
                      dist = torch.norm(e_pos - p_pos, dim=1)
                      PROX_RADIUS = 3000.0
                      bonus = torch.clamp(1.0 - (dist / PROX_RADIUS), 0.0, 1.0)
                      # Only if active blocker and target is active runner
                      
                      # Fix mask logic:
                      # We need 'e_runner_mask' to match batch?
                      # e_runner_mask is [B, 2].
                      is_e_run = self.is_runner[:, real_e] # [B]
                      
                      condition = is_block & is_e_run
                      prox_rew[condition] += bonus[condition]
                 
                 # [FIX] Apply Tau (dense_mult)
                 rewards_indiv[:, i] += prox_rew * w_prox * dense_mult * is_block.float()

        # --- RW_DENIAL (Blocker -> Deny Enemy Progress) ---
        # "The Doorman Reward": Stop enemy from moving to CP.
        # Only active if w_denial > 0 and Stage >= Duel
        
        # We assume RW_DENIAL is index 15. Check if weights match env config or generic.
        # w_denial = reward_weights[:, 15] if size > 15
        # Since we use fixed indices, we rely on caller to provide correct weights tensor size.
        
        w_denial = reward_weights[:, 15] if reward_weights.shape[1] > 15 else torch.zeros(self.num_envs, device=self.device)
        
        if self.curriculum_stage >= STAGE_DUEL_FUSED and w_denial.sum() > 0:
            for i in range(4):
                is_block = ~self.is_runner[:, i]
                if not is_block.any(): continue
                
                team = i // 2
                enemy_team = 1 - team
                # Get Enemy Runner Index
                e_indices = torch.tensor([2*enemy_team, 2*enemy_team+1], device=self.device)
                is_e_run_mask = self.is_runner[:, e_indices] # [B, 2]
                
                denial_rew = torch.zeros(self.num_envs, device=self.device)
                p_pos = self.physics.pos[:, i]

                for idx_in_team in range(2):
                    real_e_idx = e_indices[idx_in_team]
                    is_target = is_e_run_mask[:, idx_in_team] # Bool [B]
                    
                    # Only process if this enemy is a runner and I am a blocker
                    valid_pair = is_block & is_target
                    if not valid_pair.any(): continue
                    
                    # 1. Check Distance (Doorman Constraint)
                    e_pos = self.physics.pos[:, real_e_idx]
                    dist = torch.norm(e_pos - p_pos, dim=1)
                    
                    # Interact only if < 3000u (Visible range)
                    close_mask = valid_pair & (dist < 3000.0)
                    if not close_mask.any(): continue
                    
                    # --- NEW [Proposal A]: Proximity Pressure ---
                    # Instead of Penalizing Enemy Speed, we Reward Proximity.
                    # "Glued to Target" = High Reward.
                    
                    # 2. Zone Reward (Intercept)
                    # Project Vector(Enemy->Me) onto Vector(Enemy->CP)
                    e_next_cp = self.next_cp_id[:, real_e_idx]
                    batch_idx = torch.arange(self.num_envs, device=self.device)
                    cp_pos = self.checkpoints[batch_idx, e_next_cp] # [B, 2]
                    
                    vec_to_cp = cp_pos - e_pos
                    d_cp = torch.norm(vec_to_cp, dim=1, keepdim=True) + 1e-6
                    dir_to_cp = vec_to_cp / d_cp 
                    
                    vec_e_me = p_pos - e_pos
                    proj_pos = (vec_e_me * dir_to_cp).sum(dim=1)
                    
                    # Zone Score: 0.0 to 1.0 (Best at ~1500u ahead)
                    zone_reward = torch.clamp(proj_pos / 1500.0, 0.0, 1.0)
                    
                    # 3. Proximity Bias (New)
                    # Reward simply for being close (< 1500u) to maintain pressure.
                    # Max 1.0 at dist=0, 0.0 at dist=1500
                    prox_score = torch.clamp(1.0 - (dist / 1500.0), 0.0, 1.0)
                    
                    # 4. Timeout Pressure (Same as Collision)
                    e_timeout = self.timeouts[:, real_e_idx]
                    # Formula: 1.0 -> 5.0 as Time 50 -> 0
                    desperation = torch.clamp(1.0 + (50.0 - e_timeout.float()) / 10.0, 1.0, 5.0)
                    
                    # Combine:
                    # Base = (Zone * 200) + (Prox * 100) -> Max 300/step
                    # Scaled by Desperation -> Max 1500/step
                    base_denial = (zone_reward * 200.0) + (prox_score * 100.0)
                    
                    denial_rew[close_mask] += base_denial[close_mask] * desperation[close_mask]
                
                # [FIX] Apply Tau (dense_mult)
                # SCALE DOWN w_denial because it is now a large Sparse Reward (~10000).
                # We want Dense component to be ~0.5 effective weight. 
                # 0.5 / 10000 = 5e-5
                rewards_indiv[:, i] += denial_rew * (w_denial * 5e-5) * dense_mult * is_block.float()
                 
        # Decrement Role Timer
        self.role_lock_timer = torch.clamp(self.role_lock_timer - 1, min=0)
            
        # Update Roles for Next Step
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._update_roles(all_ids)
        
        # Add Metrics to Info
        infos['runner_velocity'] = runner_velocity_metric
        infos['blocker_damage'] = blocker_damage_metric

        # --- FINAL BLENDING ---
        # Hybrid Reward:
        # R_total = R_indiv + Blend(R_team, spirit)
        final_rewards = rewards_indiv + rewards_team
        
        return final_rewards, self.dones.clone(), infos

    def get_obs(self):
        """
        Produce Observation Tensor for DeepSets (Body-Frame / Ego-Centric).
        Vectorized across B environments AND 4 Pods.
        
        Returns:
            self_obs: [B, 4, 14]
            entity_obs: [B, 4, 3, 13]
            cp_obs: [B, 4, 6]
            map_obs: [B, 4, MaxCP, 2] # New
        """
        
        B = self.num_envs
        device = self.device
        
        # Norm Constants
        S_POS = 1.0 / 16000.0
        S_VEL = 1.0 / 1000.0
        
        # 1. Physics State [B, 4]
        pos = self.physics.pos # [B, 4, 2]
        vel = self.physics.vel # [B, 4, 2]
        angle = self.physics.angle # [B, 4]
        
        # Rotation Helper (Vectorized for [..., 2] and [..., 1] angle)
        def rotate_vec(v, a_deg):
            # v: [..., 2], a_deg: [...]
            rad = torch.deg2rad(a_deg).unsqueeze(-1) # [..., 1]
            c = torch.cos(rad)
            s = torch.sin(rad)
            # Rotation Matrix [[c, s], [-s, c]]
            # x' = x*c + y*s
            # y' = -x*s + y*c
            vx = v[..., 0:1]
            vy = v[..., 1:2]
            nx = vx * c + vy * s
            ny = -vx * s + vy * c
            return torch.cat([nx, ny], dim=-1) # [..., 2]

        # --- Self Features (14) ---
        # 1. Local Velocity
        # Rel to own angle
        v_local = rotate_vec(vel, angle) * S_VEL # [B, 4, 2]
        
        # 2. Target Vector
        # Gather Target Pos
        # self.next_cp_id is [B, 4]
        # self.checkpoints is [B, N_CP, 2]
        # We need efficient gather.
        # Flatten batch:
        # flat_cp_ids = self.next_cp_id.long() + torch.arange(B, device=device).unsqueeze(1) * self.num_checkpoints.unsqueeze(1) # This assumes fixed N_CP? 
        # Wait, self.checkpoints is [B, 5, 2] max?
        # Actually checkpoints tensor is padded fixed size?
        # Let's check init... self.checkpoints = torch.zeros((num_envs, max_checkpoints, 2))
        # So we can index normally.
        
        # Correct Gather:
        # We want [B, 4, 2]
        # Use Cached Batch Index
        batch_idx = self.cache_batch_idx
        target_pos = self.checkpoints[batch_idx, self.next_cp_id.long()] # [B, 4, 2]
        
        t_vec_g = target_pos - pos
        dest = torch.norm(t_vec_g, dim=-1, keepdim=True) * S_POS # [B, 4, 1]
        t_vec_l = rotate_vec(t_vec_g, angle) * S_POS # [B, 4, 2]
        
        # Alignment
        # t_vec_l is (Fwd, Right) scaled.
        # Cos = Fwd / Dist, Sin = Right / Dist
        dist_safe = dest + 1e-6
        align = t_vec_l / dist_safe # [B, 4, 2] (Cos, Sin)
        
        # Scalars
        shield = (self.physics.shield_cd.float() / 3.0).unsqueeze(-1) # [B, 4, 1]
        
        # Boost (Team Shared)
        # team 0: pods 0,1. team 1: pods 2,3.
        # boost_available is [B, 2]
        # Expand to [B, 4]
        # Use Cached Team Index
        boost = self.physics.boost_available.gather(1, self.cache_team_idx).float().unsqueeze(-1) # [B, 4, 1]        
        timeout = (self.timeouts.float() / 100.0).unsqueeze(-1)
        lap = (self.laps.float() / 3.0).unsqueeze(-1)
        leader = self.is_runner.float().unsqueeze(-1)
        v_mag = torch.norm(vel, dim=-1, keepdim=True) * S_VEL
        
        # --- Rank Calculation ---
        # Calculate Rank for Observation
        # t_vec_g is [B, 4, 2] (Global vector to target)
        dist_next_obs = torch.norm(t_vec_g, dim=-1) # [B, 4]
        
        # Score calculation (same as in step, but recomputed for obs)
        score_obs = self.laps.float() * 1000.0 + self.next_cp_id.float() * 100.0 - dist_next_obs
        s_row_obs = score_obs.unsqueeze(1) # [B, 1, 4]
        s_col_obs = score_obs.unsqueeze(2) # [B, 4, 1]
        
        # Rank: Count how many entities have a score strictly greater than mine
        ranks_obs = (s_col_obs > s_row_obs).sum(dim=2).float().unsqueeze(-1) # [B, 4, 1]
        rank_norm = ranks_obs / 3.0 # Normalize [0, 1]
        
        pad = torch.zeros_like(v_mag)

        # Assemble Self
        # [B, 4, 15] (+1 Rank, +1 Pad retained)
        # v_local: 2, t_vec_l: 2, dest: 1, align: 2, shield, boost, timeout, lap, leader, v_mag, pad, rank
        self_obs = torch.cat([
            v_local, t_vec_l, dest, align, shield, boost, timeout, lap, leader, v_mag, pad, rank_norm
        ], dim=-1)
        
        # --- Entity Features (3 x 13) ---
        # We need "Others" for each Pod.
        # Use Cached Indices
        other_indices = self.cache_other_indices
        
        # Broadcast to [B, 4, 3]
        # We can gather from pos [B, 4, 2]
        # Map: For each pod i, get 3 others.
        
        # Gather logic:
        # We want [B, 4, 3, 2] (Pos of others)
        # Expand Pos to [B, 1, 4, 2] -> [B, 4, 4, 2] ?
        # Or just fancy indexing.
        # pos[:, other_indices] works if pos is [B, 4, 2] -> result [B, 4, 3, 2]
        # (PyTorch simple indexing works on specific dim if others are :)
        # Actually `pos[:, other_indices]` might interpret as batch index?
        # Careful.
        
        # Let's perform Gather on dim 1.
        # We need index tensor of shape [B, 4, 3]
        # gather_idx = other_indices.unsqueeze(0).expand(B, -1, -1) # [B, 4, 3]
        # Gather requires index to have same dims as input except on gather dim.
        # Input [B, 4, 2]. We need to gather (dim 1). 
        # Gather index must be [B, 4, 3] -> Output [B, 4, 3, 2]? 
        # gather on dim=1, index must broadcast to [B, 4, 3, 2]? No.
        # torch.gather(input, dim, index)
        # index shape determines output shape.
        # We want to gather 2D coords.
        # Gather index must capture the last dim? No gather works on one dim.
        
        # Easier: Flatten B*4
        # But B is large.
        
        # Alternative: pos is [B, 4, 2].
        # We want o_pos [B, 4, 3, 2].
        # Use simple indexing:
        # o_pos = pos[:, other_indices] # This works mostly in numpy. In torch?
        # Let's try explicit expansion.
        o_pos = pos[:, other_indices] # [B, 4, 3, 2] assuming standard advanced indexing behavior
        o_vel = vel[:, other_indices]
        o_angle = angle[:, other_indices] # [B, 4, 3]
        o_shield = self.physics.shield_cd[:, other_indices] > 0 # [B, 4, 3]
        
        # New: Runner Flag for entities
        o_is_runner = self.is_runner[:, other_indices].float().unsqueeze(-1) # [B, 4, 3, 1]
        
        # --- Ghost Injection (Robustness) ---
        # If teammate is inactive (Stage < Team), their pos is Infinity.
        # We replace it with Ghost Data (Random nearby) to avoid "Zero Collapse" in the model.
        # This ensures the model learns to handle "Valid but Random" teammate inputs during Solo/Duel.
        
        # 1. Identify Inactive Teammates
        # Mate is index 0 in 'other_indices' (See init: 0->1, 1->0, 2->3, 3->2)
        
        # Convert config.active_pods to boolean mask for fast lookup
        is_pod_active_mask = torch.zeros(4, dtype=torch.bool, device=device)
        # Handle list vs tensor config
        active_pods_list = self.config.active_pods
        is_pod_active_mask[active_pods_list] = True
        
        mate_ids = self.cache_other_indices[:, 0] # [4]
        mate_active_mask = is_pod_active_mask[mate_ids] # [4]
        mate_inactive = ~mate_active_mask # [4]
        
        if mate_inactive.any():
             # Broadcast to Batch
             mask_b = mate_inactive.unsqueeze(0).expand(B, 4) # [B, 4]
             mask_coord = mask_b.unsqueeze(-1) # [B, 4, 1]
             
             # Generate Ghosts relative to Player (so they are "seen" in local frame)
             # Range: +/- 4000
             offsets = (torch.rand((B, 4, 2), device=device) * 8000.0) - 4000.0
             ghost_pos = pos + offsets
             
             # Overwrite Position (Index 0 of dim 2 is Teammate)
             # o_pos is [B, 4, 3, 2]
             o_pos[:, :, 0, :] = torch.where(mask_coord, ghost_pos, o_pos[:, :, 0, :])
             
             # Ghost Velocity
             ghost_vel = (torch.rand((B, 4, 2), device=device) * 2000.0) - 1000.0
             o_vel[:, :, 0, :] = torch.where(mask_coord, ghost_vel, o_vel[:, :, 0, :])
             
             # Ghost Angle
             ghost_angle = (torch.rand((B, 4), device=device) * 360.0) - 180.0
             o_angle[:, :, 0] = torch.where(mask_b, ghost_angle, o_angle[:, :, 0])
             
             # Ghost Shield (Always False)
             o_shield[:, :, 0] = torch.where(mask_b, torch.tensor(False, device=device), o_shield[:, :, 0])

             # Ghost Runner Flag (Random?)
             # Let's say Ghost is a Runner 50% of time? Or Blocker?
             # For robustness, random is fine.
             random_runner = (torch.rand((B, 4), device=device) > 0.5).float().unsqueeze(-1)
             o_is_runner[:, :, 0, :] = torch.where(mask_coord, random_runner, o_is_runner[:, :, 0, :])
        
        # Current Pod State, Expanded
        # p_pos: [B, 4, 1, 2]
        p_pos = pos.unsqueeze(2) 
        p_vel = vel.unsqueeze(2)
        p_angle = angle.unsqueeze(2) # [B, 4, 1]
        
        # Deltas
        d_pos_g = o_pos - p_pos # [B, 4, 3, 2]
        d_vel_g = o_vel - p_vel
        
        # Rotate all at once
        # p_angle is [B, 4, 1]. Need to match d_pos_g [B, 4, 3, 2].
        # Expand angle to [B, 4, 3]
        p_angle_exp = p_angle.expand(-1, -1, 3) 
        
        dp_local = rotate_vec(d_pos_g, p_angle_exp) * S_POS # [B, 4, 3, 2] (Fwd, Right)
        dv_local = rotate_vec(d_vel_g, p_angle_exp) * S_VEL
        
        # Relative Angle
        # o_angle: [B, 4, 3], p_angle: [B, 4, 1]
        # Broadcasts to [B, 4, 3]
        rel_angle = o_angle - p_angle # [B, 4, 3]
        rel_rad = torch.deg2rad(rel_angle).unsqueeze(-1)
        rel_cos = torch.cos(rel_rad) # [B, 4, 3, 1]
        rel_sin = torch.sin(rel_rad)
        
        # Dist
        dist = torch.norm(d_pos_g, dim=-1, keepdim=True) * S_POS # [B, 4, 3, 1]
        
        # Mate
        # Use Cached Mate Tensor
        is_mate = self.cache_is_mate
        
        # Shield
        o_shield_f = o_shield.float().unsqueeze(-1) # [B, 4, 3, 1]
        
        # Their Target (Relative to Me)
        # o_nid = next_cp_id gathered
        # next_cp_id [B, 4]
        # Gather for others
        o_nid = self.next_cp_id[:, other_indices] # [B, 4, 3]
        # Use cached batch index for entity gather
        o_target = self.checkpoints[self.cache_batch_idx_entity, o_nid] # [B, 4, 3, 2]
        
        # p_pos is [B, 4, 1, 2]. Broadcasts fine against [B, 4, 3, 2]
        ot_vec_g = o_target - p_pos 
        
        # Rotate using expanded p_angle
        ot_vec_l = rotate_vec(ot_vec_g, p_angle_exp) * S_POS
        
        # Padding [B, 4, 3, 2] -> Now [B, 4, 3, 1] for last pad
        # REPLACE PAD WITH RANK
        # rank_norm is [B, 4, 1] (Self Rank)
        # We gather it for others
        o_rank = rank_norm[:, other_indices] # [B, 4, 3, 1]
        
        # o_timeout gathering (New for Tier 2)
        # timeout is [B, 4, 1] (Already normalized)
        # Gather [B, 4, 3, 1]
        o_timeout = timeout[:, other_indices]
        
        # Concat Entity
        # dp(2), dv(2), cos(1), sin(1), dist(1), mate(1), shield(1), ot(2), is_runner(1), rank(1), timeout(1) -> 14
        entity_obs = torch.cat([
            dp_local, dv_local, rel_cos, rel_sin, dist, is_mate, o_shield_f, ot_vec_l, o_is_runner, o_rank, o_timeout
        ], dim=-1) # [B, 4, 3, 14]
        
        # --- CP Features (10) ---
        # Next (CP1) -> Next+1 (CP2) -> Next+2 (CP3)
        # Vector CP1->CP2 and CP2->CP3 in My Body Frame
        # CP1 is target_pos [B, 4, 2]
        
        # Get Checkpoints
        cp1_ids = self.next_cp_id.long()
        cp2_ids = (cp1_ids + 1) % self.num_checkpoints.unsqueeze(1)
        cp3_ids = (cp1_ids + 2) % self.num_checkpoints.unsqueeze(1)
        
        cp2_pos = self.checkpoints[batch_idx, cp2_ids] # [B, 4, 2]
        cp3_pos = self.checkpoints[batch_idx, cp3_ids] # [B, 4, 2]
        
        # 1. Vector CP1->CP2
        v12_g = cp2_pos - target_pos # [B, 4, 2]
        v12_l = rotate_vec(v12_g, angle) * S_POS
        
        # 2. Vector CP2->CP3
        v23_g = cp3_pos - cp2_pos
        v23_l = rotate_vec(v23_g, angle) * S_POS
        
        # 3. Global Progress (Num CPs Left)
        # Total to do = Laps * NumCPs
        # Done = LapsDone * NumCPs + NextID (approx)
        # Left = Total - Done
        # Normalized by e.g. 18 (3 laps * 6 cps)
        n_cps = self.num_checkpoints.unsqueeze(1).float() # [B, 1]
        total_cps = MAX_LAPS * n_cps
        done_cps = self.laps.float() * n_cps + self.next_cp_id.float()
        left_cps = total_cps - done_cps # [B, 4]
        
        # Scale: 0.0 to 1.0 (18.0 is typical max)
        cps_left_norm = (left_cps / 20.0).unsqueeze(-1) # [B, 4, 1]
        
        # 4. Corner Angle (Cos)
        # Angle between Me->CP1 and CP1->CP2
        # t_vec_g is Me->CP1
        # v12_g is CP1->CP2
        # Normalize
        def normalize(v):
            n = torch.norm(v, dim=-1, keepdim=True) + 1e-6
            return v / n
        
        dir_01 = normalize(t_vec_g)
        dir_12 = normalize(v12_g)
        corner_cos = (dir_01 * dir_12).sum(dim=-1, keepdim=True) # [B, 4, 1]
        
        # 5. Max Speed Heuristic (1 / Curvature?)
        # Simple heuristic: If corner is sharp (cos < 0), max speed is low.
        # If straight (cos ~ 1), max speed is high.
        # Map -1..1 to 0..1
        max_speed_factor = (corner_cos + 1.0) * 0.5 
        
        # Assemble [B, 4, 10]
        # t_vec_l(2), v12_l(2), v23_l(2), Left(1), Corner(1), Speed(1), Pad(1)
        pad_cp = torch.zeros((B, 4, 1), device=device)
        
        cp_obs = torch.cat([
            t_vec_l, v12_l, v23_l, cps_left_norm, corner_cos, max_speed_factor, pad_cp
        ], dim=-1) # [B, 4, 10]
        
        # Permute to [4, B, ...] for contiguous memory access per Pod
        # self_obs: [B, 4, 14] -> [4, B, 14]
        self_c = self_obs.permute(1, 0, 2).contiguous()
        
        # entity_obs: [B, 4, 3, 13]
        # Split into Teammate (Index 0) and Enemies (Indices 1,2)
        # Teammate: [B, 4, 1, 13] -> Squeeze -> [B, 4, 13]
        tm_obs = entity_obs[:, :, 0, :]
        en_obs = entity_obs[:, :, 1:, :] # [B, 4, 2, 13]
        
        # Permute
        tm_c = tm_obs.permute(1, 0, 2).contiguous()
        en_c = en_obs.permute(1, 0, 2, 3).contiguous()
        cp_c = cp_obs.permute(1, 0, 2).contiguous()
        
        # --- Map Features (MaxCP * 2) ---
        # All checkpoints relative to current pod pos and orientation
        # self.checkpoints: [B, N_CP, 2] (Padded)
        
        # 1. Expand checkpoints to [B, 4, N_CP, 2]
        all_cps = self.checkpoints.unsqueeze(1).expand(-1, 4, -1, -1)
        
        # 2. Global Delta
        # p_pos: [B, 4, 1, 2]
        d_map_g = all_cps - p_pos
        
        # 3. Rotate
        # p_angle_exp needs to be [B, 4, N_CP] check dims
        # p_angle [B, 4, 1] -> [B, 4, N_CP]
        max_cp = all_cps.shape[2]
        p_angle_map = p_angle.expand(-1, -1, max_cp)
        
        map_local = rotate_vec(d_map_g, p_angle_map) * S_POS # [B, 4, MaxCP, 2]
        
        # 4. Canonical Ordering
        # Rotate checkpoints so the list starts with next_cp_id?
        # This makes the sequence invariant to lap progress.
        # "Relative Map": CP[0] is always NextCP, CP[1] is Next+1, ...
        
        # Gather indices: [next, next+1, ...]
        # next_cp_id: [B, 4]
        # We need a gather index tensor of shape [B, 4, MaxCP]
        
        # range [0, 1, ... MaxCP-1]
        range_idx = torch.arange(max_cp, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, MaxCP]
        
        # num_checkpoints: [B] -> [B, 4, 1]
        n_cp_b = self.num_checkpoints.unsqueeze(1).unsqueeze(2)
        
        # next_cp_id: [B, 4] -> [B, 4, 1]
        start_idx = self.next_cp_id.long().unsqueeze(-1)
        
        # (Start + Range) % NumCheckpoints
        # Careful with padding: if n_cp < max_cp, modulo might access invalid indices if we just wrapped max_cp.
        # But here max_cp IS the tensor size.
        # Wait, self.checkpoints has valid data up to n_cp, then zeros? Or Repeated?
        # Usually tracked by n_cp.
        # For Transformer, we want the valid sequence loop.
        
        # Correct Gather Index:
        # idx = (start + range) % n_cp
        # But we must mask out indices >= n_cp if logic requires.
        # However, `self.checkpoints` likely contains 0,0 at padding.
        # If we modulo by n_cp, we stay in valid range.
        # Then we might need to mask the output if we want true Variable Length.
        # But keeping it simple: Just rotate the loop.
        
        gather_idx = (start_idx + range_idx) % n_cp_b
        
        # Expand indices for gather on dim 2
        # map_local: [B, 4, N, 2]
        # gather_idx: [B, 4, N] -> [B, 4, N, 2]
        gather_idx_2d = gather_idx.unsqueeze(-1).expand(-1, -1, -1, 2)
        
        map_ordered = torch.gather(map_local, 2, gather_idx_2d)
        
        # Masking? 
        # If we gathered valid CPs, we are good.
        # What about padding/zeros? 
        # If n_cp=3 and max=6, we have indices 0,1,2,0,1,2... repeated?
        # Ideally we want 0,1,2, PAD, PAD.
        # Mask: range_idx < n_cp_b
        mask_valid = (range_idx < n_cp_b).float().unsqueeze(-1) # [B, 4, MaxCP, 1]
        map_ordered = map_ordered * mask_valid # Zero out padding
        
        map_c = map_ordered.permute(1, 0, 2, 3).contiguous() # [4, B, N, 2]
        
        return self_c, tm_c, en_c, cp_c, map_c
