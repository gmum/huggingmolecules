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

    def load_dataset_from_csv(self, dataset_path: str) -> List[T_MoleculeEncoding]:
        data = pd.read_csv(dataset_path)
        smiles_list = data.iloc[:, 0].values
        y_list = data.iloc[:, 1].values
        return self._encode_smiles_list(smiles_list, y_list)

    def get_data_loader(self, dataset: List[T_MoleculeEncoding], *, batch_size: int, shuffle: bool = False,
                        num_workers: int = 0) -> DataLoader:
        return DataLoader(dataset, batch_size,
                          collate_fn=self._collate_encodings,
                          num_workers=num_workers,
                          shuffle=shuffle)

    def get_data_loaders(self, train_dataset: List[T_MoleculeEncoding], valid_dataset: List[T_MoleculeEncoding],
                         test_dataset: Optional[List[T_MoleculeEncoding]] = None, *, batch_size, num_workers=0) -> \
    Tuple[DataLoader, ...]:
        train_loader = self.get_data_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = self.get_data_loader(valid_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)
        test_loader = None if test_dataset is None else self.get_data_loader(test_dataset, batch_size=batch_size,
                                                                             shuffle=False, num_workers=num_workers)
        if test_dataset is not None:
            return train_loader, valid_loader, test_loader
        else:
            return train_loader, valid_loader

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> T_MoleculeEncoding:
        raise NotImplementedError

    def _collate_encodings(self, encodings: List[T_MoleculeEncoding]) -> T_BatchEncoding:
        raise NotImplementedError
