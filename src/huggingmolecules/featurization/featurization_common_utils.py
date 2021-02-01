from typing import List, Iterable

import torch

from .featurization_api import T_BatchEncoding


def one_hot_vector(value: int, choices: List[int], extra_category: bool = False) -> Iterable:
    """Converts a value to a one-hot vector based on options in lst"""
    encoding = [0] * len(choices)
    if extra_category:
        encoding.append(0)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def stack_y(encodings: List[T_BatchEncoding]):
    if any(e.y is None for e in encodings):
        return None
    else:
        return torch.stack([torch.tensor(e.y).float() for e in encodings]).unsqueeze(1)
