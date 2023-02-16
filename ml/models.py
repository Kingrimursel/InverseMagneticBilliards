import torch.nn as nn


class ReLuModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        y = self.model(x)
        return y
    
    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)
