import logging
import os
from typing import Generic, List, Type, Any

import torch
import torch.nn as nn

from ..downloading.downloading_utils import from_cache
from ..featurization.featurization_api import T_BatchEncoding, T_Config, PretrainedFeaturizerMixin


class PretrainedModelBase(nn.Module, Generic[T_BatchEncoding, T_Config]):
    def __init__(self, config: T_Config):
        super().__init__()
        self.config = config

    def forward(self, batch: T_BatchEncoding):
        raise NotImplementedError

    @classmethod
    def _get_archive_dict(cls) -> dict:
        raise NotImplementedError

    @classmethod
    def get_config_cls(cls) -> Type[T_Config]:
        raise NotImplementedError

    @classmethod
    def get_featurizer_cls(cls) -> Type[PretrainedFeaturizerMixin[Any, T_BatchEncoding, T_Config]]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls,
                        pretrained_name: str, *,
                        excluded: List[str] = None,
                        config: T_Config = None) -> "PretrainedModelBase[T_BatchEncoding, T_Config]":
        archive_dict = cls._get_archive_dict()
        file_path = from_cache(pretrained_name, archive_dict, 'pt')
        if not file_path:
            file_path = os.path.expanduser(pretrained_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)
            if not config:
                raise AttributeError('Set \'config\' attribute when using local path to weights.')
        if not config:
            config_cls = cls.get_config_cls()
            config = config_cls.from_pretrained(pretrained_name)
        model = cls(config)
        model.load_weights(file_path, excluded=excluded)
        return model

    def init_weights(self, init_type: str):
        for p in self.parameters():
            if p.dim() > 1:
                if init_type == 'uniform':
                    nn.init.xavier_uniform_(p)
                elif init_type == 'normal':
                    nn.init.xavier_normal_(p)
                else:
                    raise NotImplementedError()

    def _remove_excluded(self, dictionary: dict, *, excluded: List[str] = None):
        excluded = excluded if excluded else []
        return {k: v for k, v in dictionary.items() if all(k.split('.')[0] != p for p in excluded)}

    def load_weights(self, file_path: str, *, excluded: List[str] = None):
        state_dict = torch.load(file_path, map_location='cpu')
        state_dict = self._remove_excluded(state_dict, excluded=excluded)
        result = self.load_state_dict(state_dict, strict=False)
        if len(result.missing_keys) > 0:
            logging.info(f'Missing keys when loading: {result.missing_keys}')
        if len(result.unexpected_keys) > 0:
            logging.warning(f'Unexpected keys when loading: {result.unexpected_keys}')

    def save_weights(self, file_path: str, *, excluded: List[str] = None):
        state_dict = self.state_dict()
        state_dict = self._remove_excluded(state_dict, excluded=excluded)
        torch.save(state_dict, file_path)
