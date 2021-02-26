import os
import random
import tempfile
from typing import Type

from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from tests.common.api import AbstractTestCase


class ConfigurationApiTestBase(AbstractTestCase):
    config_cls: Type[PretrainedConfigMixin]
    config_arch_dict: dict

    def get_default_dict(self):
        return self.config_cls().to_dict()

    def random_param(self):
        params = list(self.config_cls().to_dict().keys())
        return {random.choice(params): random.random()}

    def test_set_value_in_constructor(self):
        param = self.random_param()
        config = self.config_cls(**param)
        expected = self.get_default_dict()
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

        expected = self.get_default_dict()
        expected.update(param_1)
        expected.update(param_2)

        self.test.assertEqual(config_second.to_dict(), expected)
