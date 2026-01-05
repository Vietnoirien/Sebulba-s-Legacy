
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
import copy
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.deepsets import PodAgent

def test_vmap_inference():
    print("--- Testing Vectorized Inference (vmap) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Setup Population (Size 2)
    agent1 = PodAgent().to(device)
    agent2 = PodAgent().to(device)
    
    # Monkey patch forward for vmap compatibility (functional_call invokes forward)
    agent1.forward = agent1.get_action_and_value
    agent2.forward = agent2.get_action_and_value # Just for symmetry
    
    # Make weights different to verify independence
    with torch.no_grad():
        for p in agent2.parameters():
            p.add_(0.1)
            
    # 2. Extract State
    # We need to structure params as {name: [Pop=2, ...]}
    params1 = dict(agent1.named_parameters())
    params2 = dict(agent2.named_parameters())
    
    buffers1 = dict(agent1.named_buffers())
    buffers2 = dict(agent2.named_buffers())
    
    # Stack Params
    stacked_params = {}
    for name in params1.keys():
        stacked_params[name] = torch.stack([params1[name], params2[name]])
        
    # Stack Buffers
    stacked_buffers = {}
    for name in buffers1.keys():
        stacked_buffers[name] = torch.stack([buffers1[name], buffers2[name]])
        
    # 3. Dummy Inputs [Pop=2, Batch=10, Dims...]
    batch_size = 10
    
    # Obs Dims from PodAgent code
    # Self: 15, Tm: 13, En: 13, CP: 6
    obs_s = torch.randn(2, batch_size, 15, device=device)
    obs_tm = torch.randn(2, batch_size, 13, device=device)
    obs_en = torch.randn(2, batch_size, 2, 13, device=device) # [B, N_En, Dim]
    obs_cp = torch.randn(2, batch_size, 6, device=device)
    
    # 4. Define Functional Wrapper
    # Signature must be: func(params, buffers, *args)
    def functional_forward(params, buffers, s, tm, en, cp):
        # We use agent1 as the "template" module.
        # functional_call replaces its weights with 'params' temporarily.
        return functional_call(agent1, (params, buffers), (s, tm, en, cp), kwargs={'compute_divergence': False})
        # Returns: action, log_prob, entropy, value
        
    # 5. Vectorize
    # in_dims: params=0, buffers=0, args=(0, 0, 0, 0)
    # randomness='different' allows independent sampling per batch element
    vmap_fwd = vmap(functional_forward, in_dims=(0, 0, 0, 0, 0, 0), randomness='different')
    
    # 6. Execute
    print("Executing vmap forward...")
    actions, log_probs, entropies, values = vmap_fwd(stacked_params, stacked_buffers, obs_s, obs_tm, obs_en, obs_cp)
    
    print(f"Output Shapes: Act={actions.shape}, LP={log_probs.shape}, Val={values.shape}")
    
    # 7. Verification (Manual Run)
    print("Verifying against serial execution...")
    # Run Agent 1
    a1, lp1, e1, v1 = agent1.get_action_and_value(obs_s[0], obs_tm[0], obs_en[0], obs_cp[0])
    
    # Run Agent 2 (Must use functional call or proper instance to be sure, but agent2 instance works too)
    a2, lp2, e2, v2 = agent2.get_action_and_value(obs_s[1], obs_tm[1], obs_en[1], obs_cp[1])
    
    # Compare
    # Agent 1
    diff_v1 = (values[0] - v1).abs().max()
    print(f"Agent 1 Value Diff: {diff_v1.item():.6f} (Max Val: {v1.abs().max():.6f})")
    
    # Agent 2
    diff_v2 = (values[1] - v2).abs().max()
    print(f"Agent 2 Value Diff: {diff_v2.item():.6f} (Max Val: {v2.abs().max():.6f})")
    
    # Use relative tolerance or looser absolute
    if diff_v1 < 1e-4 and diff_v2 < 1e-3:
        print(">> VERIFICATION PASSED: Inference matches serial execution (within float32 tolerance).")
    else:
        print(">> VERIFICATION FAILED: Mismatch detected.")
        exit(1)

def test_vmap_training():
    print("\n--- Testing Vectorized Training (vmap grad) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup
    agent1 = PodAgent().to(device)
    agent2 = PodAgent().to(device)
    
    # Monkey patch forward
    agent1.forward = agent1.get_action_and_value
    agent2.forward = agent2.get_action_and_value
    
    with torch.no_grad():
        for p in agent2.parameters(): p.add_(0.1)
        
    params1 = dict(agent1.named_parameters())
    params2 = dict(agent2.named_parameters())
    stacked_params = {n: torch.stack([params1[n], params2[n]]) for n in params1}
    
    buffers1 = dict(agent1.named_buffers())
    buffers2 = dict(agent2.named_buffers())
    stacked_buffers = {n: torch.stack([buffers1[n], buffers2[n]]) for n in buffers1}
    
    # Data
    B = 4
    obs_s = torch.randn(2, B, 15, device=device)
    obs_tm = torch.randn(2, B, 13, device=device)
    obs_en = torch.randn(2, B, 2, 13, device=device)
    obs_cp = torch.randn(2, B, 6, device=device)
    
    targets = torch.randn(2, B, 1, device=device) # Dummy Value targets
    
    # Generate valid hybrid actions
    # [Cont(2), Shield(1), Boost(1)]
    act_cont = torch.randn(2, B, 2, device=device)
    act_shield = torch.randint(0, 2, (2, B, 1), device=device).float()
    act_boost = torch.randint(0, 2, (2, B, 1), device=device).float()
    
    dummy_actions = torch.cat([act_cont, act_shield, act_boost], dim=2)
    # Note: DeepSets uses floats for all actions in specific tensor format
    
    # Define Loss Function per agent
    def compute_loss(params, buffers, s, tm, en, cp, action, target):
        # Forward with ACTION (No sampling)
        # We need to kwargs action=action
        # But functional_call arg passing is positional for the args tuple.
        # We wrapped it in tests.
        
        # We need to invoke get_action_and_value(..., action=action)
        # functional_call(agent, params_dict, (args...)) calls agent(*args).
        # We can pass action as a positional arg if get_action_and_value accepts it positionally or we wrap it.
        # get_action_and_value signature: (self_obs, ..., next_cp_obs, action=None)
        # So we can pass 5 positional args.
        
        _, _, _, value = functional_call(agent1, (params, buffers), (s, tm, en, cp, action), kwargs={'compute_divergence': False})
        # Dummy MSE Loss
        loss = ((value - target) ** 2).mean()
        return loss

    # Vectorize Grad
    # in_dims: params=0, buffers=0, args=(0, 0, 0, 0, 0, 0) -> 6 args (s, tm, en, cp, action, target)
    compute_grad = vmap(grad(compute_loss), in_dims=(0, 0, 0, 0, 0, 0, 0, 0))
    
    # Execute
    print("Executing vmap grad...")
    grads = compute_grad(stacked_params, stacked_buffers, obs_s, obs_tm, obs_en, obs_cp, dummy_actions, targets)
    
    # Verify
    print("Verifying Gradients...")
    
    # Agent 1 Manual
    agent1.zero_grad()
    _, _, _, v1 = agent1.get_action_and_value(obs_s[0], obs_tm[0], obs_en[0], obs_cp[0])
    l1 = ((v1 - targets[0]) ** 2).mean()
    l1.backward()
    
    # Compare Gradient of 'critic_net.backbone.0.weight'
    key = 'critic_net.backbone.0.weight'
    man_grad1 = agent1.critic_net.backbone[0].weight.grad
    vmap_grad1 = grads[key][0]
    
    diff = (man_grad1 - vmap_grad1).abs().max()
    print(f"Gradient Diff Agent 1 ({key}): {diff:.6f}")
    
    if diff < 1e-5:
        print(">> VERIFICATION PASSED: Gradients match.")
    else:
        print(">> VERIFICATION FAILED: Gradient mismatch.")
        exit(1)

if __name__ == "__main__":
    test_vmap_inference()
    test_vmap_training()
