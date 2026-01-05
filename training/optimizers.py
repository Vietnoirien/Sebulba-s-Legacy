
import torch
import math

class VectorizedAdam:
    """
    Vectorized implementation of Adam optimizer.
    Optimizes a batch of identical models with different weights (PopSize).
    Supports per-agent Learning Rates (PBT).
    """
    def __init__(self, params_dict: dict, lrs: torch.Tensor, betas=(0.9, 0.999), eps=1e-8):
        """
        params_dict: {name: Tensor[PopSize, ...]}
        lrs: Tensor[PopSize] (Initial learning rates)
        """
        self.params = params_dict
        self.lrs = lrs # Ref to tensor (should be kept updated)
        self.betas = betas
        self.eps = eps
        self.step_count = 0 
        
        # Initialize State
        self.exp_avg = {}
        self.exp_avg_sq = {}
        
        # Verify device match
        first_p = next(iter(params_dict.values()))
        self.device = first_p.device
        
        for name, p in self.params.items():
            self.exp_avg[name] = torch.zeros_like(p)
            self.exp_avg_sq[name] = torch.zeros_like(p)

    def reset_state(self):
        """Resets the optimizer state (momentum, variance, step count)."""
        self.step_count = 0
        for name in self.params:
            self.exp_avg[name].zero_()
            self.exp_avg_sq[name].zero_()

            
    def step(self, grads_dict: dict):
        with torch.no_grad():
            self.step_count += 1
            beta1, beta2 = self.betas
            
            # Bias Correction
            bias_correction1 = 1 - beta1 ** self.step_count
            bias_correction2 = 1 - beta2 ** self.step_count
            
            # Calculate Step Size factor: lr / bias_correction1
            # lrs is [Pop]. Need to broadcast to [Pop, 1, 1...] match param dim.
            # But params differ in dims.
            # We can reshape lr on the fly.
            
            for name, param in self.params.items():
                if name not in grads_dict:
                    continue
                    
                grad = grads_dict[name]
                m = self.exp_avg[name]
                v = self.exp_avg_sq[name]
                
                # Update Moments (In-place)
                # m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # v = beta2 * v + (1 - beta2) * grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Adaptive Step Size
                # denom = sqrt(v) / sqrt(bias2) + eps
                denom = (v.sqrt().div_(math.sqrt(bias_correction2))).add_(self.eps)
                
                # broadcast lr to param shape
                # param shape: [Pop, D1, D2...]
                # lr shape: [Pop]
                # view: [Pop, 1, 1...]
                target_shape = [param.shape[0]] + [1] * (param.ndim - 1)
                step_size = (self.lrs / bias_correction1).view(target_shape)
                
                # param = param - step_size * (m / denom)
                # Note: previous implementation tried addcdiv with non-scalar value which failed? 
                # Or wait, step 652 had logic. 
                # Let's use the explicit 'sub' logic I wrote in step 652.
                
                update = m.div(denom).mul_(step_size)
                param.sub_(update)

    def set_lrs(self, new_lrs: torch.Tensor):
        self.lrs.copy_(new_lrs)

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'exp_avg': self.exp_avg,
            'exp_avg_sq': self.exp_avg_sq,
            'lrs': self.lrs
        }
    
    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.exp_avg = state_dict['exp_avg']
        self.exp_avg_sq = state_dict['exp_avg_sq']
        self.lrs.copy_(state_dict['lrs'])
