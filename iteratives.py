from itertools import tee
import torch


# characteristics of periodic orbit
L = 5
m = 4
n = 4


class Orbit:
    def __init__(self, L, m, n, *args, **kwargs):
        """ Class of an orbit of the annulus

        Args:
            L (float): circumference of billiards table
            m (int): _description_
            n (int): _description_
        """
        self.m = m
        self.n = n
        self.L = L

        # initialize orbit equidistributed
        self.x = torch.linspace(0, L, n)

    def parameters(self, *args, **kwargs):
        return [self.x]

    def __call__(self, *args, **kwargs):
        return self.x


class Action(torch.autograd.Function):
    # def __init__(self, orbit, *args, **kwargs):
    #    """Discrete action class

    #    Args:
    #        orbit (Orbit): instance of orbit associated to action
    #    """

    #    self.orbit = orbit

    # @staticmethod
    # def __call__(self):
    #    x0s, x1s = tee(self.orbit())
    #    next(x1s, None)
    #    pairwise = list(zip(x0s, x1s))
    #
    #        action = 0
    #        for xs in pairwise:
    #            action += G(*xs)
    #
    #        return action

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        return 1

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        return 1


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
