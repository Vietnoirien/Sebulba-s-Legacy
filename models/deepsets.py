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
    Input: Self(15) + CP(6) = 21
    Output: Thrust(1), Angle(1)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.input_dim = 15 + 6
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

class PodActor(nn.Module):
    """
    Recurrent Actor.
    Flow: Features -> LSTM -> Heads.
    """
    def __init__(self, hidden_dim=64, lstm_hidden=64): 
        super().__init__()
        
        # 1. Feature Extractors (No Heads yet)
        # Pilot Features: Self(15) + CP(6) = 21 -> 64
        self.pilot_embed = nn.Sequential(
            layer_init(nn.Linear(21, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )
        
        # Commander Features
        self.role_embedding = nn.Embedding(2, 16)
        # CommanderNet (modified to be an embedder)
        # Input: Self(15) + Team(16) + Enemy(16) + Role(16) = 63
        # We'll use the existing CommanderNet structure but strip heads? 
        # Easier to just inline the encoder part or reuse CommanderNet class if modified.
        # Let's redefine it here to be explicit and concise for export.
        
        self.enemy_encoder = nn.Sequential(
            layer_init(nn.Linear(13, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 16))
        )
        
        self.commander_backbone = nn.Sequential(
            layer_init(nn.Linear(15 + 16 + 16 + 16, 128)), # Self(15)+Team(16)+Ctx(16)+Role(16)
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)), # Output 64 to match Pilot
            nn.ReLU()
        )
        
        # 2. LSTM Core
        # Input: Pilot(64) + Commander(64) = 128
        # 2. LSTM Core
        # Input: Pilot(64) + Commander(64) = 128
        self.lstm = CustomLSTM(input_size=128, hidden_size=lstm_hidden)
        
        # 3. Heads (From LSTM Output)
        # Pilot Heads
        self.head_thrust = layer_init(nn.Linear(lstm_hidden, 1), std=0.01)
        self.head_angle = layer_init(nn.Linear(lstm_hidden, 1), std=0.01)
        
        # Commander Heads
        self.head_shield = layer_init(nn.Linear(lstm_hidden, 2), std=0.01)
        self.head_boost = layer_init(nn.Linear(lstm_hidden, 2), std=0.01)
        
        # LogStd (Learnable)
        self.actor_logstd = nn.Parameter(torch.zeros(2))
        
    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, role_ids, lstm_state=None, done=None):
        # self_obs: [B, 15]
        # Check for sequence dimension.
        # If input is [B, S, F], we process as sequence.
        # If input is [B, F], we unsqueeze to [B, 1, F].
        
        has_seq = (self_obs.ndim == 3)
        if not has_seq:
            self_obs = self_obs.unsqueeze(1)
            teammate_obs = teammate_obs.unsqueeze(1)
            enemy_obs = enemy_obs.unsqueeze(1)
            next_cp_obs = next_cp_obs.unsqueeze(1)
            role_ids = role_ids.unsqueeze(1)
            if done is not None: done = done.unsqueeze(1)
            
        B, S, _ = self_obs.shape
        
        # --- 1. Features ---
        # Flatten for MLP: [B*S, F]
        flat_self = self_obs.reshape(B*S, -1)
        flat_cp = next_cp_obs.reshape(B*S, -1)
        flat_tm = teammate_obs.reshape(B*S, -1)
        flat_en = enemy_obs.reshape(B*S, enemy_obs.shape[2], -1)
        flat_role = role_ids.reshape(B*S)
        
        # A. Pilot Embed
        pilot_in = torch.cat([flat_self, flat_cp], dim=1)
        p_emb = self.pilot_embed(pilot_in) # [B*S, 64]
        
        # B. Commander Embed
        # Encode Teammate
        tm_lat = self.enemy_encoder(flat_tm) # [B*S, 16]
        # Encode Enemies
        # flat_en is [B*S, N, 13]
        bs_en, n_en, dim_en = flat_en.shape
        en_flat_flat = flat_en.reshape(bs_en * n_en, dim_en)
        en_lat = self.enemy_encoder(en_flat_flat).view(bs_en, n_en, -1)
        en_ctx, _ = torch.max(en_lat, dim=1) # [B*S, 16]
        
        r_emb = self.role_embedding(flat_role) # [B*S, 16]
        
        cmd_in = torch.cat([flat_self, tm_lat, en_ctx, r_emb], dim=1)
        c_emb = self.commander_backbone(cmd_in) # [B*S, 64]
        
        # --- 2. LSTM ---
        # Combine [B, S, 128]
        lstm_in = torch.cat([p_emb, c_emb], dim=1).view(B, S, -1)
        
        if lstm_state is None:
             # Initial state
             # We rely on caller to provide state, or init zero.
             # PPO rollout provides state.
             device = self_obs.device
             h0 = torch.zeros(1, B, self.lstm.hidden_size, device=device)
             c0 = torch.zeros(1, B, self.lstm.hidden_size, device=device)
             lstm_state = (h0, c0)

        # Handle DONEs (Reset state)
        # If training on sequences, we mask state where done=True?
        # Typically PPO handles this by resetting state buffers, but inside forward 
        # during inference, we might need it.
        # For simplicity, we assume state is managed externally or sequences don't cross episodes.
        
        out, (hn, cn) = self.lstm(lstm_in, lstm_state)
        # out: [B, S, H]
        
        # --- 3. Heads ---
        # Apply heads directly to out [B, S, H]
        # Linear layer accepts arbitrary leading dimensions
        
        thrust_logits = self.head_thrust(out)
        angle_input = self.head_angle(out)
        
        shield_logits = self.head_shield(out)
        boost_logits = self.head_boost(out)
        
        # Activations
        thrust_mean = torch.sigmoid(thrust_logits)
        angle_mean = torch.tanh(angle_input)
        
        if not has_seq:
            # Squeeze back [B, 1, D] -> [B, D]
            thrust_mean = thrust_mean.squeeze(1)
            angle_mean = angle_mean.squeeze(1)
            shield_logits = shield_logits.squeeze(1)
            boost_logits = boost_logits.squeeze(1)
        
            # Reshape for consistency if needed, but squeeze does it.
            
        # Unified std expansion (Dimension Agnostic)
        dist_mean = torch.cat([thrust_mean, angle_mean], dim=-1) # [..., 2]
        std = torch.exp(self.actor_logstd).expand_as(dist_mean)
             
        std = torch.clamp(std, min=0.05)
        
        return (thrust_mean, angle_mean), std, (shield_logits, boost_logits), (hn, cn)

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
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(15 + 13 + 13 + 6, 256)), # Self+Team+Enemy(Mean)+CP
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU()
        )
        self.lstm = CustomLSTM(128, lstm_hidden)
        self.value_head = layer_init(nn.Linear(lstm_hidden, 1), std=1.0)
        
    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, lstm_state=None):
        has_seq = (self_obs.ndim == 3)
        if not has_seq:
            self_obs = self_obs.unsqueeze(1)
            teammate_obs = teammate_obs.unsqueeze(1)
            enemy_obs = enemy_obs.unsqueeze(1)
            next_cp_obs = next_cp_obs.unsqueeze(1)
            
        B, S, _ = self_obs.shape
        
        # Simple Enemy Aggregation (Mean)
        en_mean = enemy_obs.mean(dim=2) # [B, S, 13]
        
        # Concat Features
        x = torch.cat([self_obs, teammate_obs, en_mean, next_cp_obs], dim=2) # [B, S, F]
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

class PodAgent(nn.Module):
    def __init__(self, hidden_dim=128, lstm_hidden=48): 
        super().__init__()
        
        self.hidden_dim = hidden_dim 
        
        # Recurrent Actor
        self.actor = PodActor(hidden_dim=hidden_dim, lstm_hidden=lstm_hidden)
        
        # Recurrent Critic
        self.critic_net = PodCritic(hidden_dim=256, lstm_hidden=lstm_hidden)
        
    def get_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, lstm_state=None):
        val, _ = self.critic_net(self_obs, teammate_obs, enemy_obs, next_cp_obs, lstm_state)
        return val#.squeeze(-1) # Squeeze last dim? handled by caller check

    def get_action_and_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, action=None, role_ids=None, actor_state=None, critic_state=None, done=None):
        # NOTE: role_ids must be passed externally or derived.
        # Derived:
        if role_ids is None:
             if self_obs.ndim == 3:
                  role_ids = (self_obs[:, :, 11] > 0.5).long()
             else:
                  role_ids = (self_obs[:, 11] > 0.5).long()

        # 1. Actor Forward
        means_pair, std, logits_pair, (hn_a, cn_a) = self.actor(self_obs, teammate_obs, enemy_obs, next_cp_obs, role_ids, actor_state, done)
        
        # 2. Critic Forward
        # Note: Critic maintains separate state? Yes ideally.
        value, (hn_c, cn_c) = self.critic_net(self_obs, teammate_obs, enemy_obs, next_cp_obs, critic_state)
        
        # Unpack
        thrust_mean, angle_mean = means_pair
        shield_logits, boost_logits = logits_pair
        
        # Construct Distributions
        dist_cont = torch.distributions.Normal(torch.cat([thrust_mean, angle_mean], -1), std)
        dist_shield = torch.distributions.Categorical(logits=shield_logits)
        dist_boost = torch.distributions.Categorical(logits=boost_logits)
        
        if action is None:
            ac_cont = dist_cont.sample()
            s = dist_shield.sample()
            b = dist_boost.sample()
            action = torch.cat([ac_cont, s.unsqueeze(-1).float(), b.unsqueeze(-1).float()], dim=-1)
        else:
            # Action provided [B, S, 4] or [B, 4]
            ac_cont = action[..., :2]
            s = action[..., 2]
            b = action[..., 3]
            
        lp_cont = dist_cont.log_prob(ac_cont).sum(-1)
        lp_s = dist_shield.log_prob(s)
        lp_b = dist_boost.log_prob(b)
        
        log_prob = lp_cont + lp_s + lp_b
        entropy = dist_cont.entropy().sum(-1) + dist_shield.entropy() + dist_boost.entropy()
        
        # Squeeze values if needed (PPO expects [B] or [B, S])
        value = value.squeeze(-1)
        
        # Return states for buffer storage
        states = {
            'actor': (hn_a, cn_a),
            'critic': (hn_c, cn_c)
        }
        
        return action, log_prob, entropy, value, states
             
    def forward(self, *args, method='get_action_and_value', **kwargs):
        if method == 'get_value':
            return self.get_value(*args, **kwargs)
        return self.get_action_and_value(*args, **kwargs)
