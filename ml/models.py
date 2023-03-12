import torch.nn as nn


class ReLuModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()

        # self.model = nn.Sequential(
        #    nn.Linear(input_dim, 10),
        #    nn.ELU(),
        #    nn.Linear(10, 10),
        #    nn.ELU(),
        #    nn.Linear(10, 10),
        #    nn.ELU(),
        #    nn.Linear(10, output_dim)
        # )

        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ELU(),
            nn.Linear(8, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.ELU(),
            nn.Linear(8, output_dim)
        )

    def forward(self, x):
        y = self.model(x)
        return y

    # def parameters(self, *args, **kwargs):
    #    return self.model.parameters(*args, **kwargs)
