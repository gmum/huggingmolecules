from abc import abstractmethod
from typing import *
import pandas as pd
from torch.utils.data import DataLoader

T_MoleculeEncoding = TypeVar('T_MoleculeEncoding')
T_BatchEncoding = TypeVar('T_BatchEncoding')


class PretrainedFeaturizerBase(Generic[T_MoleculeEncoding, T_BatchEncoding]):
    def __call__(self, smiles_list: List[str], y_list: Optional[List[float]] = None) -> T_BatchEncoding:
        encodings = self._encode_smiles_list(smiles_list, y_list)
        batch = self._collate_encodings(encodings)
        return batch

    def _encode_smiles_list(self, smiles_list: List[str], y_list: Optional[List[float]]) -> List[T_MoleculeEncoding]:
        encodings = []
        if y_list is not None:
            assert len(smiles_list) == len(y_list)
        y_list = y_list if y_list is not None else (None for _ in smiles_list)
        for smiles, y in zip(smiles_list, y_list):
            encodings.append(self._encode_smiles(smiles, y))
        return encodings

    def load_dataset_from_csv(self, dataset_path):
        data = pd.read_csv(dataset_path)
        smiles_list = data.iloc[:, 0].values
        y_list = data.iloc[:, 1].values
        return self._encode_smiles_list(smiles_list, y_list)

    def get_data_loader(self, dataset, *, batch_size, shuffle=False):
        return DataLoader(dataset, batch_size, shuffle, collate_fn=self._collate_encodings)

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> T_MoleculeEncoding:
        raise NotImplementedError

    def _collate_encodings(self, encodings: List[T_MoleculeEncoding]) -> T_BatchEncoding:
        raise NotImplementedError
