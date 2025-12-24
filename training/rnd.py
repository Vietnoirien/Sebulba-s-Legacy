import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RNDModel(nn.Module):
    """
    Random Network Distillation (RND) Module for Intrinsic Curiosity.
    Source: "Exploration by Random Network Distillation" (Burda et al., 2018)
    
    Structure:
    1. Target Network (Frozen, Randomly Initialized)
    2. Predictor Network (Trainable, tries to predict Target output)
    
    Intrinsic Reward = || Target(s) - Predictor(s) ||^2
    """
    def __init__(self, input_dim, output_dim=128, hidden_dim=256, device='cuda', lr=1e-4):
        super(RNDModel, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Target Network (Fixed)
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        
        # Predictor Network (Trainable)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        
        # Freeze Target
        for param in self.target.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        
    def forward(self, x):
        """
        Forward pass for both networks.
        x: [Batch, InputDim]
        Returns: target_feature, predictor_feature
        """
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        return target_out, pred_out
    
    def compute_intrinsic_reward(self, x):
        """
        Compute intrinsic reward for a batch of observations.
        Returns tensor [Batch]
        """
        target_out, pred_out = self.forward(x)
        
        # MSE per sample
        error = (target_out - pred_out).pow(2).sum(dim=1)
        # Scale? Usually RND rewards are 0.0-1.0 range, but MSE can varies.
        # We return raw MSE / 2
        return error / 2.0
    
    def update(self, x):
        """
        Update Predictor to match Target.
        Returns loss item.
        """
        target_out, pred_out = self.forward(x)
        loss = F.mse_loss(pred_out, target_out)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()
