import os
import tempfile
from typing import Type

from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from tests.common.api import AbstractTestCase
from tests.common.utils import get_random_config_param


class ConfigurationApiTestBase(AbstractTestCase):
    config_cls: Type[PretrainedConfigMixin]
    config_arch_dict: dict

    def get_default_dict(self):
        return self.config_cls().to_dict()

    def test_set_value_in_constructor(self):
        param = get_random_config_param(self.config_cls)
        config = self.config_cls(**param)
        expected = self.get_default_dict()
        expected.update(param)
        assert config.to_dict() == expected

    def test_save_and_load(self):
        config_first = self.config_cls(**get_random_config_param(self.config_cls))
        with tempfile.TemporaryDirectory() as tmp:
            json_file_path = os.path.join(tmp, 'config.json')
            config_first.save(json_file_path)
            config_second = self.config_cls.from_pretrained(json_file_path)

        self.test.assertEqual(config_first.to_dict(), config_second.to_dict())
