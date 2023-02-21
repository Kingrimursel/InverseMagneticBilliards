import torch.nn as nn


class ReLuModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim)
        )

    def forward(self, x):
        y = self.model(x)
        return y
    
    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)
