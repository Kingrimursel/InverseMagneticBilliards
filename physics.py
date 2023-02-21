import torch

from util import pair


class DiscreteAction:
    def __init__(self, action_fn, orbit, exact=False, *args, **kwargs):
        self.action_fn = action_fn
        self.orbit = orbit
        self.exact = exact


    def __call__(self, x):
        if self.exact:
            points = self.orbit.points(x=x)
            points2 = torch.roll(points, 1, 0)
            dists = torch.norm(points - points2, dim=1)
            action = torch.sum(dists)
        else:
            action = torch.sum(self.action_fn(pair(x)))
        return action