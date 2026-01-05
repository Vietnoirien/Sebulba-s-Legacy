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
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.self_obs_dim = 15
        self.teammate_obs_dim = 13
        self.enemy_obs_dim = 13
        self.latent_dim = 16
        self.hidden_dim = hidden_dim
        
        # Shared Encoder for Enemies/Teammate
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(self.enemy_obs_dim, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.latent_dim)),
        )
        
        # Backbone Input: Self(15) + Team(16) + Enemy(16) = 47
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(self.self_obs_dim + self.latent_dim * 2, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        )
        
        # Heads
        self.head_shield = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)
        self.head_boost = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)
        self.head_bias_thrust = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        self.head_bias_angle = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        
    def forward(self, self_obs, teammate_obs, enemy_obs):
        # 1. Encode Teammate [B, 13] -> [B, 16]
        tm_latent = self.encoder(teammate_obs)
        
        # 2. Encode Enemies [B, N, 13] -> [B, 16] (DeepSets)
        B, N, _ = enemy_obs.shape
        flat_en = enemy_obs.reshape(B*N, -1)
        enc_en = self.encoder(flat_en).view(B, N, -1)
        env_ctx, _ = torch.max(enc_en, dim=1)
        
        # 3. Backbone
        x = torch.cat([self_obs, tm_latent, env_ctx], dim=1)
        h = self.backbone(x)
        
        return (
            self.head_shield(h),
            self.head_boost(h),
            self.head_bias_thrust(h),
            self.head_bias_angle(h)
        )

class PodActor(nn.Module):
    """
    Split Backbone Actor.
    Combines Pilot (Driving) and Commander (Tactics).
    """
    def __init__(self, hidden_dim=128): # hidden_dim for Commander
        super().__init__()
        
        self.pilot = PilotNet(hidden_dim=64)
        self.commander = CommanderNet(hidden_dim=hidden_dim)
        
        # LogStd (Learnable)
        self.actor_logstd = nn.Parameter(torch.zeros(2))
        
    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs):
        # Pilot
        p_thrust, p_angle = self.pilot(self_obs, next_cp_obs)
        
        # Commander
        c_shield, c_boost, c_bias_thrust, c_bias_angle = self.commander(self_obs, teammate_obs, enemy_obs)
        
        # Combine
        # Thrust = Sigmoid(Pilot + CommanderBias)
        # Angle = Tanh(Pilot + CommanderBias)
        
        thrust_logits = p_thrust + c_bias_thrust
        angle_input = p_angle + c_bias_angle
        
        thrust_mean = torch.sigmoid(thrust_logits)
        angle_mean = torch.tanh(angle_input)
        
        # Clamp std
        std = torch.exp(self.actor_logstd).expand_as(torch.cat([thrust_mean, angle_mean], 1))
        std = torch.clamp(std, min=0.05)
        
        return (thrust_mean, angle_mean), std, (c_shield, c_boost)

class PodCritic(nn.Module):
    """
    Centralized Critic (Larger Capacity).
    Not exported, so size doesn't matter much.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.self_obs_dim = 15 # +1 for Rank
        self.teammate_obs_dim = 13
        self.enemy_obs_dim = 13
        self.next_cp_dim = 6
        self.latent_dim = 16
        self.hidden_dim = hidden_dim
        
        self.enemy_encoder = nn.Sequential(
            layer_init(nn.Linear(self.enemy_obs_dim, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.latent_dim)),
        )
        
        # Input: Self(14) + TeammateLatent(16) + EnemyLatent(16) + CP(6) = 52
        input_dim = self.self_obs_dim + self.latent_dim + self.latent_dim + self.next_cp_dim
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(input_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        )
        
        self.value_head = layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)

    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs):
        # 1. Encode Teammate
        tm_latent = self.enemy_encoder(teammate_obs)
        
        # 2. Encode Enemies
        B, N, _ = enemy_obs.shape
        flat_enemies = enemy_obs.reshape(B * N, -1)
        encodings = self.enemy_encoder(flat_enemies)
        encodings = encodings.view(B, N, -1)
        enemy_context, _ = torch.max(encodings, dim=1)
        
        # 3. Backbone
        combined = torch.cat([self_obs, tm_latent, enemy_context, next_cp_obs], dim=1)
        features = self.backbone(combined)
        return self.value_head(features)

class PodAgent(nn.Module):
    def __init__(self, hidden_dim=128): # hidden_dim applies to Commander
        super().__init__()
        
        self.hidden_dim = hidden_dim # For export reference
        
        # Dual Actors
        self.runner_actor = PodActor(hidden_dim=hidden_dim)
        self.blocker_actor = PodActor(hidden_dim=hidden_dim)
        
        # Central Critic
        self.critic_net = PodCritic(hidden_dim=256)
        
    def get_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs):
        return self.critic_net(self_obs, teammate_obs, enemy_obs, next_cp_obs)

    def get_action_and_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, action=None, compute_divergence=False):
        # 1. Critic
        value = self.critic_net(self_obs, teammate_obs, enemy_obs, next_cp_obs)
        
        # 2. Actors - Heterogeneous Routing
        # Determine Role: self_obs[..., 8] is 'leader' (Runner) flag in env.py (index 8 in assembled cat?)
        # Let's verify env.py self_obs assembly indices.
        # cat([v_local(2), t_vec_l(2), dest(1), align(2), shield(1), boost(1), timeout(1), lap(1), leader(1), v_mag(1), pad(1)])
        # 0,1 | 2,3 | 4 | 5,6 | 7 | 8 | 9 | 10 | 11 | 12 | 13
        # Leader is index 11.
        
        is_runner = self_obs[:, 11] > 0.5 # [B]
        
        # Efficiency:
        # We could run both actors on all inputs and mask results.
        # Since B=4096 and GPU is fast, this avoids branch divergence/scatter-gather overhead in Python.
        
        (r_means, r_std, r_logits) = self.runner_actor(self_obs, teammate_obs, enemy_obs, next_cp_obs)
        (b_means, b_std, b_logits) = self.blocker_actor(self_obs, teammate_obs, enemy_obs, next_cp_obs)
        
        # Masking
        # r_means tuple (thrust, angle)
        # expand mask
        mask = is_runner.unsqueeze(-1).float() # [B, 1]
        inv_mask = 1.0 - mask
        
        # Continuous
        thrust_mean = r_means[0] * mask + b_means[0] * inv_mask
        angle_mean = r_means[1] * mask + b_means[1] * inv_mask
        
        # Std (Mixed)
        std = r_std * mask.expand_as(r_std) + b_std * inv_mask.expand_as(b_std)
        
        # Discrete Logits
        # [B, 2]
        shield_logits = r_logits[0] * mask + b_logits[0] * inv_mask
        boost_logits = r_logits[1] * mask + b_logits[1] * inv_mask
        
        # Distribution Construction (Same as before)
        means = torch.cat([thrust_mean, angle_mean], dim=1)
        dist_cont = torch.distributions.Normal(means, std)
        
        dist_shield = torch.distributions.Categorical(logits=shield_logits)
        dist_boost = torch.distributions.Categorical(logits=boost_logits)
        
        # Sampling
        if action is None:
            ac_cont = dist_cont.sample()
            s = dist_shield.sample()
            b = dist_boost.sample()
            action = torch.cat([ac_cont, s.unsqueeze(1).float(), b.unsqueeze(1).float()], dim=1)
        else:
            ac_cont = action[:, :2]
            s = action[:, 2]
            b = action[:, 3]
            
        lp_cont = dist_cont.log_prob(ac_cont).sum(1)
        lp_s = dist_shield.log_prob(s)
        lp_b = dist_boost.log_prob(b)
        
        log_prob = lp_cont + lp_s + lp_b
        entropy = dist_cont.entropy().sum(1) + dist_shield.entropy() + dist_boost.entropy()
        
        divergence = torch.tensor(0.0, device=self_obs.device)
        
        if compute_divergence:
            # Construct Distributions for both heads
            # Runner
            d_r_cont = torch.distributions.Normal(torch.cat(r_means, 1), r_std)
            d_r_s = torch.distributions.Categorical(logits=r_logits[0])
            d_r_b = torch.distributions.Categorical(logits=r_logits[1])
            
            # Blocker
            d_b_cont = torch.distributions.Normal(torch.cat(b_means, 1), b_std)
            d_b_s = torch.distributions.Categorical(logits=b_logits[0])
            d_b_b = torch.distributions.Categorical(logits=b_logits[1])
            
            # KL(P || Q) + KL(Q || P) (Symmetric)
            # Continuous (Sum over dims)
            # Normal distribution KL is closed form and vmap-safe usually.
            kl_cont = torch.distributions.kl.kl_divergence(d_r_cont, d_b_cont).sum(1) + \
                      torch.distributions.kl.kl_divergence(d_b_cont, d_r_cont).sum(1)
                      
            # Discrete - SAFE IMPLEMENTATION for vmap
            # Avoid torch.distributions.kl due to boolean masking (vmap limitation)
            # KL(P||Q) = sum(p * (log_p - log_q))
            # d_x_s.logits are stored in distribution
            
            def safe_cat_kl(logits_p, logits_q):
                p = torch.softmax(logits_p, dim=-1)
                log_p = torch.log_softmax(logits_p, dim=-1)
                log_q = torch.log_softmax(logits_q, dim=-1)
                return (p * (log_p - log_q)).sum(-1)

            kl_s = safe_cat_kl(r_logits[0], b_logits[0]) + \
                   safe_cat_kl(b_logits[0], r_logits[0])
                   
            kl_b = safe_cat_kl(r_logits[1], b_logits[1]) + \
                   safe_cat_kl(b_logits[1], r_logits[1])
                   
            divergence = (kl_cont + kl_s + kl_b)
        
        if compute_divergence:
             return action, log_prob, entropy, value, divergence
        else:
             return action, log_prob, entropy, value
             
    def forward(self, *args, method='get_action_and_value', **kwargs):
        if method == 'get_value':
            return self.get_value(*args, **kwargs)
        return self.get_action_and_value(*args, **kwargs)
