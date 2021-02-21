from typing import Generic

import torch
import torch.nn as nn

from ..featurization.featurization_api import T_BatchEncoding, T_Config

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
        self._config = config

    def get_config(self) -> T_Config:
        return self._config

    def forward(self, batch: T_BatchEncoding):
        raise NotImplementedError

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def get_config_cls(cls):
        raise NotImplementedError

    @classmethod
    def get_featurizer_cls(cls):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name: str, task: str, **kwargs):
        config_cls = cls.get_config_cls()
        config = config_cls.from_pretrained(pretrained_name, **kwargs)
        model = cls(config)
        file_path = cls._get_arch_from_pretrained_name(pretrained_name)
        model._load_pretrained_weights(file_path)
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

    def _load_pretrained_weights(self, file_path: str):
        pretrained_state_dict = torch.load(file_path, map_location='cpu')
        if 'state_dict' in pretrained_state_dict:
            print(pretrained_state_dict.keys())
            print(pretrained_state_dict['args'])
            pretrained_state_dict = pretrained_state_dict['state_dict']
        if 'model' in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict['model']
        model_state_dict = self.state_dict()
        for name, param in pretrained_state_dict.items():
            if 'generator' in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            if name not in model_state_dict:
                old_name = name
                for size in [2, 3]:
                    try:
                        suff = '.'.join(name.split('.')[-size:])
                        name = name.replace(suff, mapping[suff])
                        break
                    except KeyError:
                        print(f'failed {suff}')

                print(f'{old_name} -> {name}')
            model_state_dict[name].copy_(param)

    def load_weights(self, file_path: str):
        state_dict = torch.load(file_path)
        if 'state_dict' in state_dict:
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict['state_dict'].items()}
        self.load_state_dict(state_dict)
