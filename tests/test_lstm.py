
import torch
import sys
import os
sys.path.append(os.getcwd())
from models.deepsets import PodAgent
from export import quantize_weights

def test_lstm_agent():
    print("Testing PodAgent with LSTM...")
    agent = PodAgent(hidden_dim=128, lstm_hidden=64)
    
    # 1. Test Single Step (B, F)
    B = 4
    obs = torch.randn(B, 15)
    tm = torch.randn(B, 13)
    en = torch.randn(B, 3, 13)
    cp = torch.randn(B, 6)
    role = torch.zeros(B, dtype=torch.long)
    
    print("  Forward (Single Step)...")
    action, log, ent, val, states = agent(obs, tm, en, cp, role_ids=role)
    print("    Output shapes:", action.shape, states['actor'][0].shape)
    
    assert action.shape == (B, 4)
    assert states['actor'][0].shape == (1, B, 64)
    
    # 2. Test Sequence (B, S, F)
    S = 10
    obs_s = torch.randn(B, S, 15)
    tm_s = torch.randn(B, S, 13)
    en_s = torch.randn(B, S, 3, 13)
    cp_s = torch.randn(B, S, 6)
    role_s = torch.zeros(B, S, dtype=torch.long)
    
    print("  Forward (Sequence)...")
    action_s, log_s, ent_s, val_s, states_s = agent(obs_s, tm_s, en_s, cp_s, role_ids=role_s)
    print("    Output shapes:", action_s.shape, states_s['actor'][0].shape)
    
    assert action_s.shape == (B, S, 4)
    assert states_s['actor'][0].shape == (1, B, 64)
    
    print("PodAgent LSTM Test Passed!")
    
    # 3. Test Export Quantization
    print("Testing Quantization...")
    try:
        q, s = quantize_weights(agent.actor)
        print(f"    Quantized {len(q)} weights. Scale: {s}")
    except Exception as e:
        print(f"    Quantization Failed: {e}")
        raise e
        
    print("Export Test Passed!")

if __name__ == "__main__":
    test_lstm_agent()
