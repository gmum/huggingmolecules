from typing import Tuple

import numpy as np
import torch


def pad_array(array: torch.Tensor, *, size: Tuple[int, ...], dtype: torch.dtype = torch.float) -> torch.Tensor:
    result = torch.zeros(size=size, dtype=dtype)
    slices = tuple(slice(s) for s in array.shape)
    result[slices] = array
    return result
