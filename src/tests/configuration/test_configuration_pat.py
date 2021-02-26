import unittest

from huggingmolecules import PatConfig
from huggingmolecules.configuration.configuration_pat import PAT_CONFIG_ARCH
from tests.configuration.configuration_base import ConfigurationArchTestBase, ConfigurationApiTestBase


class ConfigurationPatApiTest(ConfigurationApiTestBase, unittest.TestCase):
    config_cls = PatConfig
    config_arch_dict = PAT_CONFIG_ARCH


class ConfigurationPatArchTest(ConfigurationArchTestBase, unittest.TestCase):
    config_cls = PatConfig
    config_arch_dict = PAT_CONFIG_ARCH
