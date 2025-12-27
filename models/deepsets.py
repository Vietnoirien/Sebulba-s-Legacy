import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PodActor(nn.Module):
    """
    Specialized Actor Network for a specific role (Runner or Blocker).
    Fits within the 50k char budget (Hidden 160).
    """
    def __init__(self, hidden_dim=160):
        super().__init__()
        self.self_obs_dim = 14
        self.teammate_obs_dim = 13
        self.enemy_obs_dim = 13
        self.next_cp_dim = 6
        self.latent_dim = 16
        self.hidden_dim = hidden_dim
        
        # Enemy Encoder
        self.enemy_encoder = nn.Sequential(
            layer_init(nn.Linear(self.enemy_obs_dim, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.latent_dim)),
        )
        
        # Backbone
        input_dim = self.self_obs_dim + self.teammate_obs_dim + self.latent_dim + self.next_cp_dim
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(input_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        )
        
        # Heads
        self.actor_thrust_mean = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        self.actor_angle_mean  = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        
        # LogStd (Learnable)
        self.actor_logstd = nn.Parameter(torch.zeros(2)) 
        
        # Discrete
        self.actor_shield = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)
        self.actor_boost = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)

    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs):
        # Enemy DeepSets
        B, N, _ = enemy_obs.shape
        flat_enemies = enemy_obs.reshape(B * N, -1)
        encodings = self.enemy_encoder(flat_enemies)
        encodings = encodings.view(B, N, -1)
        enemy_context, _ = torch.max(encodings, dim=1) 
        
        combined = torch.cat([self_obs, teammate_obs, enemy_context, next_cp_obs], dim=1)
        features = self.backbone(combined)
        
        # Actor
        thrust_mean = torch.sigmoid(self.actor_thrust_mean(features))
        angle_mean = torch.tanh(self.actor_angle_mean(features))
        std = torch.exp(self.actor_logstd).expand_as(torch.cat([thrust_mean, angle_mean], 1))
        
        shield_logits = self.actor_shield(features)
        boost_logits = self.actor_boost(features)
        
        return (thrust_mean, angle_mean), std, (shield_logits, boost_logits)

class PodCritic(nn.Module):
    """
    Centralized Critic (Larger Capacity).
    Not exported, so size doesn't matter much.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.self_obs_dim = 14
        self.teammate_obs_dim = 13
        self.enemy_obs_dim = 13
        self.next_cp_dim = 6
        self.latent_dim = 16
        self.hidden_dim = hidden_dim
        
        # Should structurally match Actor for feature compatibility?
        # Ideally yes, but we can make it deeper.
        
        self.enemy_encoder = nn.Sequential(
            layer_init(nn.Linear(self.enemy_obs_dim, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.latent_dim)),
        )
        
        input_dim = self.self_obs_dim + self.teammate_obs_dim + self.latent_dim + self.next_cp_dim
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(input_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        )
        
        self.value_head = layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)

    def forward(self, self_obs, teammate_obs, enemy_obs, next_cp_obs):
        B, N, _ = enemy_obs.shape
        flat_enemies = enemy_obs.reshape(B * N, -1)
        encodings = self.enemy_encoder(flat_enemies)
        encodings = encodings.view(B, N, -1)
        enemy_context, _ = torch.max(encodings, dim=1)
        
        combined = torch.cat([self_obs, teammate_obs, enemy_context, next_cp_obs], dim=1)
        features = self.backbone(combined)
        return self.value_head(features)

class PodAgent(nn.Module):
    def __init__(self, hidden_dim=160): # hidden_dim applies to Actors
        super().__init__()
        
        self.hidden_dim = hidden_dim # For export reference
        
        # Dual Actors
        self.runner_actor = PodActor(hidden_dim=hidden_dim)
        self.blocker_actor = PodActor(hidden_dim=hidden_dim)
        
        # Central Critic
        self.critic_net = PodCritic(hidden_dim=256)
        
    def get_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs):
        return self.critic_net(self_obs, teammate_obs, enemy_obs, next_cp_obs)

    def get_action_and_value(self, self_obs, teammate_obs, enemy_obs, next_cp_obs, action=None):
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
        
        return action, log_prob, entropy, value
