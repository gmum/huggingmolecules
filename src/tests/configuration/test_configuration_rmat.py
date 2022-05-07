import unittest

from huggingmolecules import RMatConfig
from huggingmolecules.configuration.configuration_rmat import RMAT_CONFIG_ARCH
from tests.configuration.configuration_base import ConfigurationApiTestBase


class ConfigurationRMatApiTest(ConfigurationApiTestBase, unittest.TestCase):
    config_cls = RMatConfig
    config_arch_dict = RMAT_CONFIG_ARCH
