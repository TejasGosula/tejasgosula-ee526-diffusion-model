import torch
from model import NoisePredictor
from diffusion import Diffusion

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NoisePredictor(2).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

diffusion = Diffusion(T=500)

x = torch.randn(1, 2).to(device)
for t in reversed(range(diffusion.T)):
    t_tensor = torch.tensor([t], device=device)
    x = diffusion.p_sample(model, x, t_tensor)

print("Generated sample:", x.cpu().detach().numpy())
