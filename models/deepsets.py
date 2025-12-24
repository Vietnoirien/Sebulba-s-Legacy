import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PodAgent(nn.Module):
    def __init__(self, hidden_dim=240):
        super().__init__()
        
        self.self_obs_dim = 14
        self.entity_obs_dim = 13
        self.next_cp_dim = 6
        self.latent_dim = 16
        self.hidden_dim = hidden_dim
        
        # Shared Encoder
        # Use Orthogonal Init (std=sqrt(2) for ReLU)
        self.entity_encoder = nn.Sequential(
            layer_init(nn.Linear(self.entity_obs_dim, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.latent_dim)),
        )
        
        # Backbone (Features)
        input_dim = self.self_obs_dim + self.latent_dim + self.next_cp_dim
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(input_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        )
        
        # Actor Heads
        # Use Normal Distribution with Sigmoid/Tanh means to match Export Template
        # std=0.01 for Actor means
        self.actor_thrust_mean = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        self.actor_angle_mean  = layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        
        # Learnable LogStd (State Independent) to allow exploration decay
        # Init to 0.0 (std=1.0) or -0.5 (std=0.6)
        self.actor_logstd = nn.Parameter(torch.zeros(2)) 
        
        # Discrete Heads (Logits)
        self.actor_shield = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)
        self.actor_boost = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)
        
        # Critic
        self.critic = layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)

    def _get_features(self, self_obs, entity_obs, next_cp_obs):
        B, N, _ = entity_obs.shape
        flat_entities = entity_obs.reshape(B * N, -1)
        encodings = self.entity_encoder(flat_entities)
        encodings = encodings.view(B, N, -1)
        global_context, _ = torch.max(encodings, dim=1) # Symmetric MaxPool
        
        combined = torch.cat([self_obs, global_context, next_cp_obs], dim=1)
        return self.backbone(combined)

    def get_value(self, self_obs, entity_obs, next_cp_obs):
        features = self._get_features(self_obs, entity_obs, next_cp_obs)
        return self.critic(features)

    def get_action_and_value(self, self_obs, entity_obs, next_cp_obs, action=None):
        features = self._get_features(self_obs, entity_obs, next_cp_obs)
        
        # 1. Critic
        value = self.critic(features)
        
        # 2. Actor Distributions
        
        # Means
        # Thrust: Sigmoid (0..1)
        thrust_mean = torch.sigmoid(self.actor_thrust_mean(features))
        
        # Angle: Tanh (-1..1) 
        # Note: In Env we multiply by 18. Here we just output normalized range.
        angle_mean = torch.tanh(self.actor_angle_mean(features))
        
        # Std
        std = torch.exp(self.actor_logstd).expand_as(torch.cat([thrust_mean, angle_mean], 1))
        
        # Distribution
        means = torch.cat([thrust_mean, angle_mean], dim=1)
        dist_cont = torch.distributions.Normal(means, std)
        
        # Discrete
        shield_logits = self.actor_shield(features)
        boost_logits = self.actor_boost(features)
        dist_shield = torch.distributions.Categorical(logits=shield_logits)
        dist_boost = torch.distributions.Categorical(logits=boost_logits)
        
        # Sampling
        if action is None:
            ac_cont = dist_cont.sample()
            s = dist_shield.sample()
            b = dist_boost.sample()
            
            # Combine
            action = torch.cat([ac_cont, s.unsqueeze(1).float(), b.unsqueeze(1).float()], dim=1)
            
        else:
            ac_cont = action[:, :2]
            s = action[:, 2]
            b = action[:, 3]
            
        # Log Probs
        lp_cont = dist_cont.log_prob(ac_cont).sum(1)
        lp_s = dist_shield.log_prob(s)
        lp_b = dist_boost.log_prob(b)
        
        log_prob = lp_cont + lp_s + lp_b
        entropy = dist_cont.entropy().sum(1) + dist_shield.entropy() + dist_boost.entropy()
        
        return action, log_prob, entropy, value
