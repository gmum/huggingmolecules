from typing import Tuple

import numpy as np
import torch
from torch.utils.data import random_split


def pad_array(array: torch.Tensor, *, size: Tuple[int, ...], dtype: torch.dtype = torch.float) -> torch.Tensor:
    result = torch.zeros(size=size, dtype=dtype)
    slices = tuple(slice(s) for s in array.shape)
    result[slices] = array
    return result


def split_data_random(dataset, train_size, test_size=0.0, seed=None):
    train_len = int(train_size * len(dataset))
    test_len = int(test_size * len(dataset))
    val_len = len(dataset) - train_len - test_len
    generator = torch.Generator().manual_seed(seed) if seed else None
    train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len], generator=generator)
    return train_data, val_data, test_data
