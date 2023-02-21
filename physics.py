import torch

from util import pair


class DiscreteAction:
    def __init__(self, action_fn, *args, **kwargs):
        self.action_fn = action_fn

    def __call__(self, x):
        action = torch.sum(self.action_fn(pair(x)))
        return action