from abc import abstractmethod
from typing import *

T_MoleculeEncoding = TypeVar('T_MoleculeEncoding')
T_BatchEncoding = TypeVar('T_BatchEncoding')


class PretrainedFeaturizerBase(Generic[T_MoleculeEncoding, T_BatchEncoding]):
    def __call__(self, smiles_list: List[str], y_list: Optional[List[float]] = None) -> T_BatchEncoding:
        encodings = self._encode_smiles_list(smiles_list, y_list)
        batch = self._get_batch_from_encodings(encodings)
        return batch

    def _encode_smiles_list(self, smiles_list: List[str], y_list: Optional[List[float]]) -> List[T_MoleculeEncoding]:
        encodings = []
        if y_list:
            assert len(smiles_list) == len(y_list)
        y_list = y_list if y_list else (None for _ in smiles_list)
        for smiles, y in zip(smiles_list, y_list):
            encodings.append(self._encode_smiles(smiles, y))
        return encodings

    @abstractmethod
    def _encode_smiles(self, smiles: str, y: Optional[float]) -> T_MoleculeEncoding:
        raise NotImplementedError

    @abstractmethod
    def _get_batch_from_encodings(self, encodings: List[T_MoleculeEncoding]) -> T_BatchEncoding:
        raise NotImplementedError
