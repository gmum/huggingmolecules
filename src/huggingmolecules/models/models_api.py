from typing import TypeVar, Generic

import torch
import torch.nn as nn

from ..configuration.configuration_api import PretrainedConfigMixin
from ..featurization.featurization_api import T_BatchEncoding

T_Config = TypeVar("T_Config", bound=PretrainedConfigMixin)


class PretrainedModelBase(nn.Module, Generic[T_BatchEncoding, T_Config]):
    def __init__(self, config: T_Config):
        super().__init__()
        self._config = config

    def get_config(self) -> T_Config:
        return self._config

    def forward(self, batch: T_BatchEncoding):
        raise NotImplementedError

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def _get_config_cls(cls):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        file_path = cls._get_arch_from_pretrained_name(pretrained_name)
        config_cls = cls._get_config_cls()
        config = config_cls.from_pretrained(pretrained_name)
        model = cls(config)
        model.load_weights(file_path)
        return model

    def load_weights(self, file_path: str):
        pretrained_state_dict = torch.load(file_path)
        if 'model' in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict['model']
        model_state_dict = self.state_dict()
        for name, param in pretrained_state_dict.items():
            if 'generator' in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

    def save_weights(self):
        pass
