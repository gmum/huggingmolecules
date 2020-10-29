import unittest
from dataclasses import dataclass

from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from tests.configuration.configuration_base import ConfigurationApiTestBase

MOCKED_CONFIG_ARCH = {}


@dataclass
class MockedConfig(PretrainedConfigMixin):
    int_val: int = 10
    float_val: float = 0.1
    bool_val: bool = True
    str_val: str = 'whatever'
    none_val: int = None

    @classmethod
    def _get_archive_dict(cls) -> dict:
        return MOCKED_CONFIG_ARCH


class ConfigurationApiTest(ConfigurationApiTestBase, unittest.TestCase):
    config_cls = MockedConfig
    config_arch_dict = MOCKED_CONFIG_ARCH

    def test_dict(self):
        config = self.config_cls()
        expected = {
            'int_val': 10,
            'float_val': 0.1,
            'bool_val': True,
            'str_val': 'whatever',
            'none_val': None
        }
        self.test.assertEqual(config.to_dict(), expected)
