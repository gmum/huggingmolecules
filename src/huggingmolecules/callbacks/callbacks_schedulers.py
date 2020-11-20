from torch.optim.lr_scheduler import LambdaLR

from torch.optim import Adam
import torch.optim


class IdentityScheduler(LambdaLR):
    def __init__(self, optimizer):
        super().__init__(optimizer, lambda epoch: 1)


class FancyScheduler(LambdaLR):
    def __init__(self, optimizer, model_dim, warmup):
        def lambda1(epoch):
            epoch += 1
            return 100 * (model_dim ** (-.5)) * min(epoch ** (-.5), epoch * (1e-6 + warmup) ** (-1.5))

        super().__init__(optimizer, lambda1)
