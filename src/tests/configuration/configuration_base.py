import os
import random
import tempfile
import unittest
from typing import Type

from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin


class ConfigurationArchTestBase:
    config_cls: Type[PretrainedConfigMixin]
    config_arch_dict: dict
    test: unittest.TestCase

    def setUp(self):
        self.test = self

    def test_parameters_coverage(self):
        config_first = self.config_cls()
        dict_first = config_first.to_dict()
        for pretrained_name in self.config_arch_dict.keys():
            dict_second = self.config_cls._load_dict_from_pretrained(pretrained_name)
            self.test.assertEqual(dict_first.keys(), dict_second.keys())


class ConfigurationApiTestBase:
    config_cls: Type[PretrainedConfigMixin]
    config_arch_dict: dict
    test: unittest.TestCase

    def setUp(self):
        self.test = self

    @property
    def default_dict(self):
        return self.config_cls().to_dict()

    def random_param(self):
        params = list(self.config_cls().to_dict().keys())
        return {random.choice(params): random.random()}

    def test_set_value_in_constructor(self):
        param = self.random_param()
        config = self.config_cls(**param)
        expected = self.default_dict
        expected.update(param)
        assert config.to_dict() == expected

    def test_save_and_load(self):
        config_first = self.config_cls(**self.random_param())
        with tempfile.TemporaryDirectory() as tmp:
            json_file_path = os.path.join(tmp, 'config.json')
            config_first.save(json_file_path)
            config_second = self.config_cls.from_pretrained(json_file_path)

        self.test.assertEqual(config_first.to_dict(), config_second.to_dict())

    def test_from_pretrained(self):
        config_first = self.config_cls(**self.random_param())
        with tempfile.TemporaryDirectory() as tmp:
            json_file_path = os.path.join(tmp, 'pretrained_config.json')
            config_first.save(json_file_path)
            self.config_arch_dict['existing_config'] = json_file_path
            config_second = self.config_cls.from_pretrained('existing_config')
            del self.config_arch_dict['existing_config']

        self.test.assertEqual(config_first.to_dict(), config_second.to_dict())

    def test_from_pretrained_kwargs(self):
        param_1 = self.random_param()
        config_first = self.config_cls(**param_1)
        with tempfile.TemporaryDirectory() as tmp:
            json_file_path = os.path.join(tmp, 'pretrained_config.json')
            config_first.save(json_file_path)
            self.config_arch_dict['existing_config'] = json_file_path
            param_2 = self.random_param()
            config_second = self.config_cls.from_pretrained('existing_config', **param_2)
            del self.config_arch_dict['existing_config']

        expected = self.default_dict
        expected.update(param_1)
        expected.update(param_2)

        self.test.assertEqual(config_second.to_dict(), expected)
