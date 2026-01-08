import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.deepsets import PodAgent

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = 'cpu'
    agent = PodAgent().to(device)
    
    total_params = count_parameters(agent)
    actor_params = count_parameters(agent.actor)
    critic_params = count_parameters(agent.critic_net)
    
    print(f"--- Model Complexity Report ---")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"-----------------------------")
    print(f"Actor Network: {actor_params:,}")
    print(f"  - Pilot Embed: {count_parameters(agent.actor.pilot_embed):,}")
    print(f"  - Map Encoder: {count_parameters(agent.actor.map_encoder):,}")
    print(f"  - Teammate Encoder: {count_parameters(agent.actor.teammate_encoder):,}")
    print(f"  - Enemy Encoder: {count_parameters(agent.actor.enemy_encoder):,}")
    print(f"  - Role Embedding: {count_parameters(agent.actor.role_embedding):,}")
    print(f"  - Commander Backbone: {count_parameters(agent.actor.commander_backbone):,}")
    print(f"  - LSTM Core: {count_parameters(agent.actor.lstm):,}")
    print(f"  - Heads: {count_parameters(agent.actor.head_thrust) + count_parameters(agent.actor.head_angle) + count_parameters(agent.actor.head_shield) + count_parameters(agent.actor.head_boost):,}")
    
    print(f"-----------------------------")
    print(f"Critic Network: {critic_params:,}")
    print(f"  - Encoder: {count_parameters(agent.critic_net.encoder):,}")
    print(f"  - Map Encoder: {count_parameters(agent.critic_net.map_encoder):,}")
    print(f"  - LSTM: {count_parameters(agent.critic_net.lstm):,}")
    print(f"  - Value Head: {count_parameters(agent.critic_net.value_head):,}")

if __name__ == "__main__":
    main()
