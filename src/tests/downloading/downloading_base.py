from typing import Type, List

import torch
from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from huggingmolecules.downloading.downloading_utils import get_cache_filepath
from huggingmolecules.models.models_api import PretrainedModelBase
from tests.common.api import AbstractTestCase
from tests.common.utils import get_excluded_params, get_random_config_param


class ConfigurationArchTestBase(AbstractTestCase):
    config_cls: Type[PretrainedConfigMixin]
    config_arch_dict: dict

    def test_from_pretrained(self):
        for pretrained_name in self.config_arch_dict.keys():
            dict_first = self.config_cls.from_pretrained(pretrained_name).to_dict()
            params = get_random_config_param(self.config_cls)
            dict_second = self.config_cls.from_pretrained(pretrained_name, **params).to_dict()
            dict_first.update(**params)
            self.test.assertEqual(dict_first, dict_second)

    def test_parameters_coverage(self):
        config_first = self.config_cls()
        dict_first = config_first.to_dict()
        for pretrained_name in self.config_arch_dict.keys():
            dict_second = self.config_cls._load_dict_from_pretrained(pretrained_name)
            self.test.assertEqual(dict_first.keys(), dict_second.keys())


class ModelsArchTestBase:
    model_cls: Type[PretrainedModelBase]
    model_arch_dict: dict
    head_layers: List[str]

    def test_pretrained_arch(self):
        for pretrained_name in self.model_arch_dict.keys():
            model = self.model_cls.from_pretrained(pretrained_name)

            weights_path = get_cache_filepath(pretrained_name, self.model_arch_dict, extension='pt')

            pretrained_params_set = set(torch.load(weights_path, map_location='cpu').keys())
            model_params_set = set(model.state_dict().keys())
            excluded_params_set = set(get_excluded_params(model, self.head_layers))

            assert pretrained_params_set.isdisjoint(excluded_params_set)
            assert pretrained_params_set.union(excluded_params_set) == model_params_set
