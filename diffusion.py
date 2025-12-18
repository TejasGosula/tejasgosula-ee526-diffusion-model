import torch

def linear_beta_schedule(T):
    return torch.linspace(1e-4, 0.02, T)

class Diffusion:
    def __init__(self, T=500):
        self.T = T
        self.beta = linear_beta_schedule(T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise):
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().unsqueeze(1)
        sqrt_one_minus = (1 - self.alpha_bar[t]).sqrt().unsqueeze(1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def p_sample(self, model, x, t):
        noise_pred = model(x, t)
        alpha = self.alpha[t].unsqueeze(1)
        alpha_bar = self.alpha_bar[t].unsqueeze(1)
        beta = self.beta[t].unsqueeze(1)
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        return (1 / alpha.sqrt()) * (x - beta / (1 - alpha_bar).sqrt() * noise_pred) + beta.sqrt() * noise
