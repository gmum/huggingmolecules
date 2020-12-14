import logging
import os
import pickle
from abc import abstractmethod
from typing import *
import pandas as pd
from torch.utils.data import DataLoader


class BatchEncodingProtocol(Protocol):
    @property
    def y(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


T_MoleculeEncoding = TypeVar('T_MoleculeEncoding')
T_BatchEncoding = TypeVar('T_BatchEncoding', bound=BatchEncodingProtocol)


class PretrainedFeaturizerMixin(Generic[T_MoleculeEncoding, T_BatchEncoding]):
    def __call__(self, smiles_list: List[str], y_list: Optional[List[float]] = None) -> T_BatchEncoding:
        encodings = self.encode_smiles_list(smiles_list, y_list)
        batch = self._collate_encodings(encodings)
        return batch

    def encode_smiles_list(self, smiles_list: List[str], y_list: Optional[List[float]]) -> List[T_MoleculeEncoding]:
        encodings = []
        if y_list is not None:
            assert len(smiles_list) == len(y_list)
        y_list = y_list if y_list is not None else (None for _ in smiles_list)
        for smiles, y in zip(smiles_list, y_list):
            encodings.append(self._encode_smiles(smiles, y))
        return encodings

    def _load_dataset_from_cache(self, cache_path: str) -> List[T_MoleculeEncoding]:
        logging.info(f"Loading encodings stored at '{cache_path}'")
        encodings = pickle.load(open(cache_path, "rb"))
        return encodings

    def _save_dataset_to_cache(self, dataset: List[T_MoleculeEncoding], cache_path: str):
        logging.info(f"Saving encodings to '{cache_path}'")
        pickle.dump(dataset, open(cache_path, "wb"))

    def load_dataset_from_csv(self, dataset_path: str, *, cache: bool = False) -> List[T_MoleculeEncoding]:
        cache_path = f'{dataset_path}-{type(self).__name__}-cached'
        if cache and os.path.exists(cache_path):
            return self._load_dataset_from_cache(cache_path)
        data = pd.read_csv(dataset_path)
        smiles_list = data.iloc[:, 0].values
        y_list = data.iloc[:, 1].values
        dataset = self.encode_smiles_list(smiles_list, y_list)
        if cache:
            self._save_dataset_to_cache(dataset, cache_path)
        return dataset

    def get_data_loader(self, dataset: List[T_MoleculeEncoding], *, batch_size: int, shuffle: bool = False,
                        num_workers: int = 0) -> DataLoader:
        return DataLoader(dataset, batch_size,
                          collate_fn=self._collate_encodings,
                          num_workers=num_workers,
                          shuffle=shuffle)

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> T_MoleculeEncoding:
        raise NotImplementedError

    def _collate_encodings(self, encodings: List[T_MoleculeEncoding]) -> T_BatchEncoding:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        # TODO do we need featurizer pretrained?
        return cls()
