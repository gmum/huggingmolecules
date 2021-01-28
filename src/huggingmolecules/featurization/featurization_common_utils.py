from typing import List, Iterable


def one_hot_vector(val: int, lst: List[int]) -> Iterable:
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)