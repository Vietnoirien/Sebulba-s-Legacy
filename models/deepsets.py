import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PilotNet(nn.Module):
    """
    Pilot: Robust Driving (Self + CP).
    Small capacity, focused on path following.
    Input: Self(15) + CP(10) = 25
    Output: Thrust(1), Angle(1)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.input_dim = 15 + 10
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU()
        )
        
        # Pilot Heads
        self.head_thrust = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        self.head_angle = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        
    def forward(self, self_obs, next_cp_obs):
        # [B, 21]
        x = torch.cat([self_obs, next_cp_obs], dim=1)
        h = self.net(x)
        return self.head_thrust(h), self.head_angle(h)

class CommanderNet(nn.Module):
    """
    Commander: Tactics (Self + Team + Enemy).
    Input: Self(15) + Team(13) + Enemy(13xN)
    Output: Shield(2), Boost(2), Bias_Thrust(1), Bias_Angle(1)
    """
    def __init__(self, hidden_dim=128, extra_input_dim=0):
        super().__init__()
        self.self_obs_dim = 15
        self.teammate_obs_dim = 13
        self.enemy_obs_dim = 13
        self.latent_dim = 16
        self.hidden_dim = hidden_dim
        self.extra_input_dim = extra_input_dim
        
        # Shared Encoder for Enemies/Teammate
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(self.enemy_obs_dim, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.latent_dim)),
        )
        
        # Backbone Input: Self(15) + Team(16) + Enemy(16) + Extra(N)
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(self.self_obs_dim + self.latent_dim * 2 + extra_input_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        )
        
        # Heads
        self.head_shield = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)
        self.head_boost = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)
        self.head_bias_thrust = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        self.head_bias_angle = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        
    def forward(self, self_obs, teammate_obs, enemy_obs, extra_emb=None):
        # 1. Encode Teammate [B, 13] -> [B, 16]
        tm_latent = self.encoder(teammate_obs)
        
        # 2. Encode Enemies [B, N, 13] -> [B, 16] (DeepSets)
        B, N, _ = enemy_obs.shape
        flat_en = enemy_obs.reshape(B*N, -1)
        enc_en = self.encoder(flat_en).view(B, N, -1)
        env_ctx, _ = torch.max(enc_en, dim=1)
        
        # 3. Backbone
        if extra_emb is not None:
            x = torch.cat([self_obs, tm_latent, env_ctx, extra_emb], dim=1)
        else:
            x = torch.cat([self_obs, tm_latent, env_ctx], dim=1)
            
        h = self.backbone(x)
        
        return (
            self.head_shield(h),
            self.head_boost(h),
            self.head_bias_thrust(h),
            self.head_bias_thrust(h),
            self.head_bias_angle(h)
        )

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, x, state):
        # x: [B, S, I]
        h, c = state
        h = h.squeeze(0) # [B, H]
        c = c.squeeze(0)
        
        B, S, _ = x.shape
        outputs = []
        
        # Precompute Input Projection for all steps [B, S, 4H]
        wx = self.ih(x)
        
        for t in range(S):
            gates = wx[:, t] + self.hh(h)
            # Use dim=-1 for safety (feature dim)
            i, f, g, o = gates.chunk(4, dim=-1)
            
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            
            c = f * c + i * g
            h = o * torch.tanh(c)
            
            outputs.append(h)
            
        out = torch.stack(outputs, dim=1)
        return out, (h.unsqueeze(0), c.unsqueeze(0))

class MapTransformer(nn.Module):
    """
    Compact Transformer for Map Encoding.
    Input: Sequence of Checkpoints (Relative) [B, S, 2]
    Output: Map Embedding [B, 32]
    """
    def __init__(self, input_dim=2, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Linear Projection
        self.input_proj = layer_init(nn.Linear(input_dim, d_model))
        
        # 2. Transformer Encoder
        # batch_first=True for [B, S, F]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Aggregation (Max Pool or Attention Pooling?)
        # Max Pool is simpler and robust.
        
    def forward(self, map_obs):
        # map_obs: [B, S, 2]
        x = F.relu(self.input_proj(map_obs))
        # Force Math Attention to avoid vmap warning/performance drop with efficient implementation
        # Using new context manager
        import torch.nn.attention as attention
        # backend=attention.SDPBackend.MATH
        with attention.sdpa_kernel(attention.SDPBackend.MATH):
            x = self.transformer(x) # [B, S, d_model]
        
        # Global Max Pooling
        # Masking? We assume fixed size map for now or pad with 0. 
        # DeepSets logic: Max over S dim.
        emb, _ = torch.max(x, dim=1) # [B, d_model]
        return emb


class PodCritic(nn.Module):
    """
    Centralized Critic (Recurrent).
    Does getting value need history? YES. 
    Ideally Critic uses same LSTM backbone or separate one.
    To save compute, maybe share backbone? But Separate is stable.
    Let's make Critic Recurrent too.
    """
    def __init__(self, hidden_dim=256, lstm_hidden=64):
        super().__init__()
        # Simplified feature extraction for Critic (MLP -> LSTM)
        # Simplified feature extraction for Critic (MLP -> LSTM)
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(15 + 14 + 14 + 10 + 32, 256)), # Self+Team+Enemy(Mean)+CP(10)+Map(32) = 85
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU()
        )
        self.map_encoder = MapTransformer(d_model=32, nhead=2, num_layers=1)
        self.lstm = CustomLSTM(128, lstm_hidden)
        self.value_head = layer_init(nn.Linear(lstm_hidden, 1), std=1.0)
        
    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs=None, lstm_state=None):
        has_seq = (self_obs.ndim == 3)
        if not has_seq:
            self_obs = self_obs.unsqueeze(1)
            teammate_obs = teammate_obs.unsqueeze(1)
            enemy_obs = enemy_obs.unsqueeze(1)
            next_cp_obs = next_cp_obs.unsqueeze(1)
            
        B, S, _ = self_obs.shape
        
        # Simple Enemy Aggregation (Mean)
        en_mean = enemy_obs.mean(dim=2) # [B, S, 13]
        
        # Critic also needs map?
        # For now, let's keep Critic simplified. 
        # Map Encoding
        if map_obs is None:
             B_map = B
             S_map = S if has_seq else 1
             map_obs = torch.zeros((B_map, S_map, 6, 2), device=self_obs.device)
             
        # Flatten time dim for transformer: [B*S, N_CP, 2]
        if has_seq:
            map_flat = map_obs.view(B*S, -1, 2)
        else:
            map_flat = map_obs.view(B, -1, 2) # [B, N, 2]
            
        map_emb = self.map_encoder(map_flat) # [B*S, 32] or [B, 32]
        
        if has_seq:
            map_emb = map_emb.view(B, S, 32)
        else:
             map_emb = map_emb.view(B, 1, 32)

        # Concat Features
        x = torch.cat([self_obs, teammate_obs, en_mean, next_cp_obs, map_emb], dim=2) # [B, S, F]
        flat_x = x.reshape(B*S, -1)
        
        emb = self.encoder(flat_x).view(B, S, -1)
        
        if lstm_state is None:
             device = self_obs.device
             h0 = torch.zeros(1, B, self.lstm.hidden_size, device=device)
             c0 = torch.zeros(1, B, self.lstm.hidden_size, device=device)
             lstm_state = (h0, c0)
             
        out, (hn, cn) = self.lstm(emb, lstm_state)
        
        val = self.value_head(out) # [B, S, 1]
        
        if not has_seq:
            val = val.view(B, 1)
        else:
            val = val.view(B, S, 1)
            
        return val, (hn, cn)

class PodActor(nn.Module):
    """
    Recurrent Actor with Hard-Gated Mixture of Experts (MoE).
    Trigger: Role ID (0=Runner, 1=Blocker).
    """
    def __init__(self, hidden_dim=64, lstm_hidden=32): 
        super().__init__()
        
        self.hidden_dim = hidden_dim 
        
        # 1. Feature Extractors (Split)
        # Pilot Features: Self(15) + CP(10) = 25 -> 64
        # RUNNER PILOT
        self.pilot_embed_runner = nn.Sequential(
            layer_init(nn.Linear(25, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU()
        )
        # BLOCKER PILOT
        self.pilot_embed_blocker = nn.Sequential(
            layer_init(nn.Linear(25, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU()
        )
        
        # Map Encoder (Shared)
        self.map_encoder = MapTransformer(input_dim=2, d_model=32, nhead=2, num_layers=1)
        
        # Commander Features (Shared Encoders)
        self.role_embedding = nn.Embedding(2, 16)
        
        # Teammate Encoder
        self.teammate_encoder = nn.Sequential(
            layer_init(nn.Linear(14, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 16))
        )
        
        # Enemy Encoder (Split)
        # [NEW] Prediction Head added internally? No, we decouple it for now.
        # But prediction head needs "Per Enemy" embedding.
        # So we split the encoder:
        # A. Per-Enemy Embedder
        # RUNNER ENEMY ENC
        self.enemy_embedder_runner = nn.Sequential(
            layer_init(nn.Linear(14, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 16))
        )
        # BLOCKER ENEMY ENC
        self.enemy_embedder_blocker = nn.Sequential(
            layer_init(nn.Linear(14, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 16))
        )
        # B. Prediction Head (Auxiliary) [16 -> 2] (Thrust, Angle)
        # Predicts NEXT ACTION or NEXT STATE delta?
        # Plan says: Predict Next State (Pos/Vel). But let's predict delta-pos (2) for simplicity.
        self.enemy_pred_head = layer_init(nn.Linear(16, 2), std=1.0) # Delta Pos
        
        # === MOE SPLIT ===
        
        # EXPERT 1: RUNNER
        # Input: Self(15) + Team(16) + EnemyCtx(16) + Role(16) + Map(32) = 95
        self.runner_backbone = nn.Sequential(
            layer_init(nn.Linear(95, 64)), # Hidden 64 (was 96)
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)), # Output 32 (was 48)
            nn.ReLU()
        )
        self.runner_lstm = CustomLSTM(input_size=64, hidden_size=lstm_hidden) # Pilot(32) + Run(32)
        
        # EXPERT 2: BLOCKER
        self.blocker_backbone = nn.Sequential(
            layer_init(nn.Linear(95, 64)), # Hidden 64
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)), # Output 32
            nn.ReLU()
        )
        self.blocker_lstm = CustomLSTM(input_size=64, hidden_size=lstm_hidden) # Pilot(32) + Blk(32)
        
        # Shared Heads (Action Space is same)
        # Input from LSTM (lstm_hidden)
        self.head_thrust = layer_init(nn.Linear(lstm_hidden, 1), std=0.01)
        self.head_angle = layer_init(nn.Linear(lstm_hidden, 1), std=0.01)
        self.head_shield = layer_init(nn.Linear(lstm_hidden, 2), std=0.01)
        self.head_boost = layer_init(nn.Linear(lstm_hidden, 2), std=0.01)
        
        # LogStd (Learnable)
        self.actor_logstd = nn.Parameter(torch.zeros(2))
        
        # Recurrent Critic (Kept as is, or use shared? Independent for stability)

        
    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs, role_ids, actor_state=None, done=None):
        # actor_state is now ((h_run, c_run), (h_blk, c_blk))
        # shape [1, B, H] each.
        
        has_seq = (self_obs.ndim == 3)
        if not has_seq:
            self_obs_in = self_obs.unsqueeze(1)
            teammate_obs_in = teammate_obs.unsqueeze(1)
            enemy_obs_in = enemy_obs.unsqueeze(1)
            next_cp_obs_in = next_cp_obs.unsqueeze(1)
            map_obs_in = map_obs.unsqueeze(1)
            role_ids_in = role_ids.unsqueeze(1)
            if done is not None: done_in = done.unsqueeze(1)
            else: done_in = None
        else:
            self_obs_in = self_obs
            teammate_obs_in = teammate_obs
            enemy_obs_in = enemy_obs
            next_cp_obs_in = next_cp_obs
            map_obs_in = map_obs
            role_ids_in = role_ids
            done_in = done

        B, S, _ = self_obs_in.shape
        
        # --- 1. Shared Feature Extraction ---
        # Flatten [B*S, F]
        flat_self = self_obs_in.reshape(B*S, -1)
        flat_cp = next_cp_obs_in.reshape(B*S, -1)
        flat_tm = teammate_obs_in.reshape(B*S, -1)
        flat_en = enemy_obs_in.reshape(B*S, enemy_obs_in.shape[2], -1)
        flat_role = role_ids_in.reshape(B*S)
        
        # Map
        bs_map, s_map, max_cp, dim_map = map_obs_in.shape
        flat_map = map_obs_in.reshape(bs_map * s_map, max_cp, dim_map)
        
        # A. Pilot Embed (Split)
        pilot_in = torch.cat([flat_self, flat_cp], dim=1)
        pe_run = self.pilot_embed_runner(pilot_in) # [B*S, 32]
        pe_blk = self.pilot_embed_blocker(pilot_in) # [B*S, 32]
        
        # B. Map Embed
        m_emb = self.map_encoder(flat_map) # [B*S, 32]
        
        # C. Shared Context
        tm_lat = self.teammate_encoder(flat_tm) # [B*S, 16]
        r_emb = self.role_embedding(flat_role) # [B*S, 16]
        
        # D. Enemy Embed & Prediction (Split)
        bs_en, n_en, dim_en = flat_en.shape
        en_flat_flat = flat_en.reshape(bs_en * n_en, dim_en)
        
        # Run BOTH
        en_lat_run = self.enemy_embedder_runner(en_flat_flat) # [B*S*N, 16]
        en_lat_blk = self.enemy_embedder_blocker(en_flat_flat) # [B*S*N, 16]
        
        # [AUX] Prediction (Shared Head, Dual Input)
        pred_delta_run = self.enemy_pred_head(en_lat_run)
        pred_delta_blk = self.enemy_pred_head(en_lat_blk)
        pred_delta = (pred_delta_run + pred_delta_blk) / 2.0
        
        # Pool
        en_lat_run_grouped = en_lat_run.view(bs_en, n_en, -1)
        en_lat_blk_grouped = en_lat_blk.view(bs_en, n_en, -1)
        
        en_ctx_run, _ = torch.max(en_lat_run_grouped, dim=1) # [B*S, 16]
        en_ctx_blk, _ = torch.max(en_lat_blk_grouped, dim=1) # [B*S, 16]
        
        # --- 2. MoE Switch (Scatter-Gather) ---
        # Construct Input for each Backbone
        # cmd_in_run: uses pe_run, en_ctx_run
        # cmd_in_blk: uses pe_blk, en_ctx_blk
        # Shape: Self(15) + Team(16) + Enemy(16) + Role(16) + Map(32) = 95
        # Role embedding is still helpful? Yes, keep it.
        
        cmd_in_run = torch.cat([flat_self, tm_lat, en_ctx_run, r_emb, m_emb], dim=1) # [B*S, 95]
        cmd_in_blk = torch.cat([flat_self, tm_lat, en_ctx_blk, r_emb, m_emb], dim=1) # [B*S, 95]
        
        # Masks
        # role 0: Runner, 1: Blocker
        is_runner = (flat_role == 0)
        is_blocker = (flat_role == 1)
        
        # Initialize Output Buffers
        # VMAP-Safe Execution: Run BOTH on full batch, then Mask
        out_run = self.runner_backbone(cmd_in_run)
        out_blk = self.blocker_backbone(cmd_in_blk)
        
        # Select
        mask_run = is_runner.float().unsqueeze(-1)
        mask_blk = is_blocker.float().unsqueeze(-1)
        
        # Embeddings for LSTM
        # Pilot(32) + Backbone(32) -> 64
        # We must CONCATENATE, not ADD.
        lstm_in_run = torch.cat([pe_run, out_run], dim=1).view(B, S, -1) # [B, S, 64]
        lstm_in_blk = torch.cat([pe_blk, out_blk], dim=1).view(B, S, -1) # [B, S, 64]
        
        # Unpack States
        if actor_state is None:
             device = self_obs.device
             h_run = torch.zeros(1, B, self.runner_lstm.hidden_size, device=device)
             c_run = torch.zeros(1, B, self.runner_lstm.hidden_size, device=device)
             h_blk = torch.zeros(1, B, self.blocker_lstm.hidden_size, device=device)
             c_blk = torch.zeros(1, B, self.blocker_lstm.hidden_size, device=device)
        else:
             (h_run, c_run), (h_blk, c_blk) = actor_state

        # Run BOTH LSTMs (State consistency strategy)
        out_run_lstm, (hn_run, cn_run) = self.runner_lstm(lstm_in_run, (h_run, c_run))
        out_blk_lstm, (hn_blk, cn_blk) = self.blocker_lstm(lstm_in_blk, (h_blk, c_blk))
        
        # --- Output Gating ---
        # Select output based on role_ids [B, S]
        mask_blk = role_ids_in.float().unsqueeze(-1)
        mask_run = 1.0 - mask_blk
        
        out_fused = out_run_lstm * mask_run + out_blk_lstm * mask_blk
        

        
        # --- Heads ---
        thrust_logits = self.head_thrust(out_fused)
        angle_input = self.head_angle(out_fused)
        shield_logits = self.head_shield(out_fused)
        boost_logits = self.head_boost(out_fused)
        
        # Activations
        thrust_mean = torch.sigmoid(thrust_logits)
        angle_mean = torch.tanh(angle_input)
        
        if not has_seq:
            thrust_mean = thrust_mean.squeeze(1)
            angle_mean = angle_mean.squeeze(1)
            shield_logits = shield_logits.squeeze(1)
            boost_logits = boost_logits.squeeze(1)

        dist_mean = torch.cat([thrust_mean, angle_mean], dim=-1) # [..., 2]
        std = torch.exp(self.actor_logstd).expand_as(dist_mean)
        std = torch.clamp(std, min=0.05)
        
        # Prediction Output (Aux)
        # Flattened [B*S*N, 2]. We might need to reshape it back to [B, S, N, 2] for loss
        # Or return as is?
        # Caller expects specific tuple. 
        # Let's attach it to 'states' (bad practice) or return extra?
        # Function signature is fixed? No, we can modify returns.
        pred_out = pred_delta.view(B, S, n_en, 2)
        if not has_seq: pred_out = pred_out.squeeze(1)
        
        # Return composite state
        next_state = ((hn_run, cn_run), (hn_blk, cn_blk))
        
        # Attach prediction to 'states' dict for convenient passing to loss
        # Hacky but working
        aux_out = {'pred': pred_out}

        return (thrust_mean, angle_mean), std, (shield_logits, boost_logits), next_state, aux_out 

    def get_action_and_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs=None, action=None, role_ids=None, actor_state=None, critic_state=None, done=None):
        # Allow PodActor to function standalone if needed, but primarily used via PodAgent
        pass

# NEW PodAgent
class PodAgent(nn.Module):
    def __init__(self, hidden_dim=64, lstm_hidden=32):
        super().__init__()
        self.actor = PodActor(hidden_dim, lstm_hidden)
        # Critic now uses same hidden size as Actor (32 via config)
        self.critic = PodCritic(hidden_dim=256, lstm_hidden=lstm_hidden) 

    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs=None, action=None, role_ids=None, actor_state=None, critic_state=None, done=None, method='get_action_and_value', lstm_state=None):
        if method == 'get_value':
            return self.get_value(self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs, lstm_state)
        return self.get_action_and_value(self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs, action, role_ids, actor_state, critic_state, done)
    
    def get_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs, critic_state):
        return self.critic(self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs, critic_state)

    def get_action_and_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs=None, action=None, role_ids=None, actor_state=None, critic_state=None, done=None):
        # Derived Role
        if role_ids is None:
             if self_obs.ndim == 3:
                  # Fix: Index 11 is 'Leader' (is_runner).
                  # We want role_id=0 for Runner, role_id=1 for Blocker.
                  # Logic: If is_runner (1.0) -> role_id should be 0.
                  #        If not is_runner (0.0) -> role_id should be 1.
                  # So condition: NOT (self_obs > 0.5)
                  role_ids = (self_obs[:, :, 11] < 0.5).long()
             else:
                  role_ids = (self_obs[:, 11] < 0.5).long()
        
        if map_obs is None:
             B = self_obs.shape[0]
             # Dummy map
             if self_obs.ndim == 3:
                 S = self_obs.shape[1]
                 map_obs = torch.zeros((B, S, 6, 2), device=self_obs.device)
             else:
                 map_obs = torch.zeros((B, 6, 2), device=self_obs.device)

        # 1. Actor
        means_pair, std, logits_pair, next_actor_state, aux_out = self.actor(self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs, role_ids, actor_state, done)
        
        # 2. Critic
        value, next_critic_state = self.critic(self_obs, teammate_obs, enemy_obs, next_cp_obs, map_obs, critic_state)
        
        thrust_mean, angle_mean = means_pair
        shield_logits, boost_logits = logits_pair
        
        dist_cont = torch.distributions.Normal(torch.cat([thrust_mean, angle_mean], -1), std)
        dist_shield = torch.distributions.Categorical(logits=shield_logits)
        dist_boost = torch.distributions.Categorical(logits=boost_logits)
        
        if action is None:
            ac_cont = dist_cont.sample()
            s = dist_shield.sample()
            b = dist_boost.sample()
            action = torch.cat([ac_cont, s.unsqueeze(-1).float(), b.unsqueeze(-1).float()], dim=-1)
        else:
            ac_cont = action[..., :2]
            s = action[..., 2]
            b = action[..., 3]
            
        lp_cont = dist_cont.log_prob(ac_cont).sum(-1)
        lp_s = dist_shield.log_prob(s)
        lp_b = dist_boost.log_prob(b)
        
        log_prob = lp_cont + lp_s + lp_b
        entropy = dist_cont.entropy().sum(-1) + dist_shield.entropy() + dist_boost.entropy()
        
        value = value.squeeze(-1)
        
        states = {
            'actor': next_actor_state,
            'critic': next_critic_state
        }
        
        return action, log_prob, entropy, value, states, aux_out



