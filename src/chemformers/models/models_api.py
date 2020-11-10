from typing import *

import torch
import torch.nn as nn

from src.chemformers.configuration.configuration_api import PretrainedConfigMixin

T_Config = TypeVar("T_Config", bound=PretrainedConfigMixin)
T_Model = TypeVar("T_Model")


class PretrainedModelMixin(nn.Module, Generic[T_Config, T_Model]):
    def __init__(self, config: T_Config):
        super().__init__()

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def _get_config_cls(cls) -> Type[T_Config]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name: str) -> T_Model:
        file_path = cls._get_arch_from_pretrained_name(pretrained_name)
        config_cls = cls._get_config_cls()
        config = config_cls.from_pretrained(pretrained_name)
        model = cls(config)
        model.load_weights(file_path)
        return model

    def load_weights(self, file_path: str):
        pretrained_state_dict = torch.load(file_path)
        model_state_dict = self.state_dict()
        for name, param in pretrained_state_dict.items():
            if 'generator' in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

    def save_weights(self):
        pass
