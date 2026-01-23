import torch # type: ignore

class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, t):
        """
        Takes clean x_0 and adds noise according to timestep t.
        Returns: x_t (noisy), noise (epsilon)
        """
        noise = torch.randn_like(x_0)
        
        # Get alpha_bar for the given timesteps
        # t is a batch of indices, e.g. [50, 999, 12, ...]
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().view(-1, 1)
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1)
        
        # Reparameterization Trick
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        # We need to broadcast t to the shape of x_0 (which is [Num_Nodes, 3])
        # Note: t usually comes in as [Batch_Size], we need to map it to [Num_Nodes]
        # But for simplicity in Week 1, we assume all nodes in a graph have same t
        # We'll handle this broadcasting inside the training loop.
        
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise
