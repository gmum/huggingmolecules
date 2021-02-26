import unittest

from huggingmolecules import MatConfig
from huggingmolecules.configuration.configuration_mat import MAT_CONFIG_ARCH
from tests.configuration.configuration_base import ConfigurationApiTestBase


class ConfigurationMatApiTest(ConfigurationApiTestBase, unittest.TestCase):
    config_cls = MatConfig
    config_arch_dict = MAT_CONFIG_ARCH
