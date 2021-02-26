import os
from typing import Generic, List, Type, Any

import torch
import torch.nn as nn

from ..featurization.featurization_api import T_BatchEncoding, T_Config, PretrainedFeaturizerMixin

mapping = {
    'norm.a_2': 'norm.weight',
    'norm.b_2': 'norm.bias',
    'W_1.weight': 'linears.0.weight',
    'W_1.bias': 'linears.0.bias',
    'W_2.weight': 'linears.1.weight',
    'W_2.bias': 'linears.1.bias',
    'linears.0.weight': 'linear_layers.0.weight',
    'linears.0.bias': 'linear_layers.0.bias',
    'linears.1.weight': 'linear_layers.1.weight',
    'linears.1.bias': 'linear_layers.1.bias',
    'linears.2.weight': 'linear_layers.2.weight',
    'linears.2.bias': 'linear_layers.2.bias',
    'linears.3.weight': 'output_linear.weight',
    'linears.3.bias': 'output_linear.bias',
}


class PretrainedModelBase(nn.Module, Generic[T_BatchEncoding, T_Config]):
    def __init__(self, config: T_Config):
        super().__init__()
        self.config = config

    def forward(self, batch: T_BatchEncoding):
        raise NotImplementedError

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
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
        file_path = cls._get_arch_from_pretrained_name(pretrained_name)
        if not file_path:
            file_path = pretrained_name
            if not os.path.exists(pretrained_name):
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
        self.load_state_dict(state_dict, strict=False)

    def save_weights(self, file_path: str, *, excluded: List[str] = None):
        state_dict = self.state_dict()
        state_dict = self._remove_excluded(state_dict, excluded=excluded)
        torch.save(state_dict, file_path)
