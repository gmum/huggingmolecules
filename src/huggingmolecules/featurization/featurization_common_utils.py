from typing import List, Optional, Union

import torch

from .featurization_api import T_BatchEncoding


def one_hot_vector(value: Union[float, int],
                   choices: List[Union[float, int]],
                   extra_category: bool = False) -> List[Union[float, int]]:
    encoding = [0] * len(choices)
    if extra_category:
        encoding.append(0)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def stack_y(encodings: List[T_BatchEncoding]) -> Optional[torch.FloatTensor]:
    return stack_y_list([e.y for e in encodings])


def stack_y_list(y_list: List[float]) -> Optional[torch.FloatTensor]:
    if any(y is None for y in y_list):
        return None
    else:
        return torch.stack([torch.tensor(y).float() for y in y_list]).unsqueeze(1)
