from typing import Generic, List, Optional, Type
from typing import TypeVar

import torch
from torch.utils.data import DataLoader

from ..configuration.configuration_api import PretrainedConfigMixin


class BatchEncodingProtocol:
    @property
    def y(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


T_MoleculeEncoding = TypeVar('T_MoleculeEncoding')
T_BatchEncoding = TypeVar('T_BatchEncoding', bound=BatchEncodingProtocol)
T_Config = TypeVar("T_Config", bound=PretrainedConfigMixin)


class PretrainedFeaturizerMixin(Generic[T_MoleculeEncoding, T_BatchEncoding, T_Config]):
    def __init__(self, config: T_Config):
        self.config = config

    def __call__(self, smiles_list: List[str], y_list: Optional[List[float]] = None) -> T_BatchEncoding:
        encodings = self.encode_smiles_list(smiles_list, y_list)
        batch = self._collate_encodings(encodings)
        return batch

    def encode_smiles_list(self, smiles_list: List[str], y_list: Optional[List[float]] = None) \
            -> List[T_MoleculeEncoding]:
        encodings = []
        if y_list is not None:
            assert len(smiles_list) == len(y_list)
        y_list = y_list if y_list is not None else (None for _ in smiles_list)
        for smiles, y in zip(smiles_list, y_list):
            encodings.append(self._encode_smiles(smiles, y))
        return encodings

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
    def _get_config_cls(cls) -> Type[T_Config]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name: str) \
            -> "PretrainedFeaturizerMixin(Generic[T_MoleculeEncoding, T_BatchEncoding, T_Config])":
        config_cls = cls._get_config_cls()
        config = config_cls.from_pretrained(pretrained_name)
        return cls(config)


class RecursiveToDeviceMixin:
    def _apply_to(self, value, device):
        if value is None:
            return value
        elif isinstance(value, RecursiveToDeviceMixin):
            return value.to(device)
        elif torch.is_tensor(value):
            return value.to(device)
        elif isinstance(value, (tuple, list)):
            return [self._apply_to(item, device) for item in value]
        elif isinstance(value, dict):
            return {k: self._apply_to(v, device) for k, v in value.items()}
        else:
            return value

    def to(self, device):
        for key, value in self.__dict__.items():
            self.__dict__[key] = self._apply_to(value, device)
        return self
