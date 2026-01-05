
import torch
from torch.func import functional_call, vmap
from models.deepsets import PodAgent

def test_kl_vmap_safety():
    print("Testing KL Divergence vmap safety...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Agent
    agent = PodAgent().to(device)
    params = dict(agent.named_parameters())
    buffers = dict(agent.named_buffers())
    
    # 2. Setup Batch
    batch_size = 10
    # Obs dims: Self(15), Team(13), Enemy(2, 13), CP(6)
    s = torch.randn(batch_size, 15, device=device)
    tm = torch.randn(batch_size, 13, device=device)
    en = torch.randn(batch_size, 2, 13, device=device)
    cp = torch.randn(batch_size, 6, device=device)
    
    # 3. Define Functional Wrapper
    def f_forward(params, buffers, s, tm, en, cp):
        # Call with compute_divergence=True to trigger the KL code path
        return functional_call(agent, (params, buffers), (s, tm, en, cp), kwargs={'compute_divergence': True})
    
    # 4. Run vmap
    try:
        # We need to simulate the "Population" dimension for vmap if we want to mimic PPO
        # PPO vmaps over the Population dim, but Agent is usually same structure.
        # Actually PPO vmaps over (params, buffers) where dim 0 is population.
        
        # Let's stack params to mimic population of size 2
        pop_size = 2
        stacked_params = {k: v.unsqueeze(0).repeat(pop_size, *([1]*v.ndim)) for k, v in params.items()}
        stacked_buffers = {k: v.unsqueeze(0).repeat(pop_size, *([1]*v.ndim)) for k, v in buffers.items()}
        
        # Inputs also need to match vmap dim usually, or be broadcasted.
        # In PPO: vmap(func, (0, 0, 0, 0, 0, 0)) over params and inputs.
        # Inputs are [Pop, Batch, Dim].
        s_pop = s.unsqueeze(0).repeat(pop_size, 1, 1)
        tm_pop = tm.unsqueeze(0).repeat(pop_size, 1, 1)
        en_pop = en.unsqueeze(0).repeat(pop_size, 1, 1, 1)
        cp_pop = cp.unsqueeze(0).repeat(pop_size, 1, 1)
        
        print(f"Running vmap on device {device} with Pop={pop_size}, Batch={batch_size}...")
        
        # Run it!
        # Returns: action, log_prob, entropy, value, divergence
        # randomness='different' is critical because agents sample actions
        res = vmap(f_forward, randomness='different')(stacked_params, stacked_buffers, s_pop, tm_pop, en_pop, cp_pop)
        
        # Unpack
        divergence = res[4]
        
        print("Success! vmap returned.")
        print(f"Divergence shape: {divergence.shape}")
        print(f"Divergence values: {divergence}")
        
        if torch.isnan(divergence).any():
            print("FAILURE: NaNs detected in divergence.")
            return False
            
        return True
        
    except Exception as e:
        print(f"FAILURE: vmap crashed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_kl_vmap_safety():
        print("TEST PASSED")
    else:
        print("TEST FAILED")
        exit(1)
