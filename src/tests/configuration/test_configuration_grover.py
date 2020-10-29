import unittest

from huggingmolecules import GroverConfig
from huggingmolecules.configuration.configuration_grover import GROVER_CONFIG_ARCH
from tests.configuration.configuration_base import ConfigurationApiTestBase


class ConfigurationGroverApiTest(ConfigurationApiTestBase, unittest.TestCase):
    config_cls = GroverConfig
    config_arch_dict = GROVER_CONFIG_ARCH
