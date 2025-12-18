import torch
import math

def sample_toy_data(n):
    theta = torch.rand(n) * 2 * math.pi
    r = 1.0
    x = r * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return x


## Data

This project uses synthetic toy datasets generated programmatically for visualization and experimentation.
No external data download is required.

Optionally, MNIST can be used via torchvision.datasets for image-based experiments.
