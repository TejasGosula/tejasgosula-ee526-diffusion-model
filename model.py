import torch
import torch.nn as nn

class NoisePredictor(nn.Module):
    def __init__(self, data_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        t = t.unsqueeze(1).float()
        x = torch.cat([x, t], dim=1)
        return self.net(x)
