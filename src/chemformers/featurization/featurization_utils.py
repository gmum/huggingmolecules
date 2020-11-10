from typing import Tuple

import numpy as np


def pad_array(array: np.ndarray, *, shape: Tuple[int, ...], dtype: type = np.float32) -> np.ndarray:
    result = np.zeros(shape, dtype=dtype)
    slices = tuple(slice(s) for s in array.shape)
    result[slices] = array
    return result