import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-4, device='cuda'):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape, device=device))
        self.register_buffer('var', torch.ones(shape, device=device))
        self.register_buffer('count', torch.tensor(epsilon, device=device))
        self.shape = shape
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def forward(self, x, fixed=False):
        if self.training and not fixed:
            self.update(x)
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)
