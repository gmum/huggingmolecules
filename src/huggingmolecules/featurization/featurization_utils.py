import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data import Subset, Dataset

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


def split_data_from_file(dataset: Dataset, split_path):
    split = np.load(split_path, allow_pickle=True)
    train_split, val_split, test_split = split.tolist()
    train_data = Subset(dataset, train_split)
    val_data = Subset(dataset, val_split)
    test_data = Subset(dataset, test_split)
    return train_data, val_data, test_data
