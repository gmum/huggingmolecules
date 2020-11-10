from typing import *
from abc import abstractmethod

import torch
import numpy as np

T_MoleculeEncoding = TypeVar('T_MoleculeEncoding')
T_BatchEncoding = TypeVar('T_BatchEncoding')


class PretrainedFeaturizerBase(Generic[T_MoleculeEncoding, T_BatchEncoding]):
    def __call__(self, smiles_list: List[str]) -> T_BatchEncoding:
        encodings = self._encode(smiles_list)
        batch = self._get_batch_from_encodings(encodings)
        return batch

    @abstractmethod
    def _encode(self, smiles_list: List[str]) -> List[T_MoleculeEncoding]:
        raise NotImplementedError

    @abstractmethod
    def _get_batch_from_encodings(self, encodings: List[T_MoleculeEncoding]) -> T_BatchEncoding:
        raise NotImplementedError


def pad_array(array: np.ndarray, *, shape: Tuple[int, ...], dtype: type = np.float32) -> np.ndarray:
    result = np.zeros(shape, dtype=dtype)
    slices = tuple(slice(s) for s in array.shape)
    result[slices] = array
    return result
