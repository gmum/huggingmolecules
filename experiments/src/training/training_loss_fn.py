import torch
from torch.nn import functional as F


class RMSELoss:
    def __call__(self, input, target):
        return torch.sqrt(F.mse_loss(input, target))
