import unittest

from huggingmolecules import MatppConfig
from huggingmolecules.configuration.configuration_matpp import MATPP_CONFIG_ARCH
from tests.configuration.configuration_base import ConfigurationApiTestBase


class ConfigurationMatppApiTest(ConfigurationApiTestBase, unittest.TestCase):
    config_cls = MatppConfig
    config_arch_dict = MATPP_CONFIG_ARCH
