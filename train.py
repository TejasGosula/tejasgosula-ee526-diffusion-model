import torch
from model import NoisePredictor
from diffusion import Diffusion
from utils import sample_toy_data

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dim = 2
model = NoisePredictor(data_dim).to(device)
diffusion = Diffusion(T=500)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(5000):
    x0 = sample_toy_data(128).to(device)
    t = torch.randint(0, diffusion.T, (x0.size(0),), device=device)
    noise = torch.randn_like(x0)

    xt = diffusion.q_sample(x0, t, noise)
    noise_pred = model(xt, t)

    loss = ((noise - noise_pred) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")
