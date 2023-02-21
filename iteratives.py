from itertools import tee
import torch



def gradient_descent(orbit, n_epochs=100):
    pass

class DiscreteAction:
    def __init__(self, orbit, *args, **kwargs):
        self.orbit = orbit

        self.action = Action.apply

    def __call__(self):
        x0s, x1s = tee(self.orbit())
        next(x1s, None)
        pairwise = list(zip(x0s, x1s))

        action = 0
        for xs in pairwise:
            action += self.action(xs)

        return action


orbit = Orbit(L, m, n)
action = DiscreteAction(orbit)

optimizer = torch.optim.SGD(orbit.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()

action.backward()
optimizer.step()
