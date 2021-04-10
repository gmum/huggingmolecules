import unittest

from huggingmolecules import MatppConfig, MatppModel
from huggingmolecules.configuration.configuration_matpp import MATPP_CONFIG_ARCH
from huggingmolecules.models.models_matpp import MATPP_MODEL_ARCH
from tests.downloading.downloading_base import ConfigurationArchTestBase, ModelsArchTestBase


class ConfigurationMatppArchTest(ConfigurationArchTestBase, unittest.TestCase):
    config_cls = MatppConfig
    config_arch_dict = MATPP_CONFIG_ARCH


class ModelsMatppArchTest(ModelsArchTestBase, unittest.TestCase):
    model_cls = MatppModel
    model_arch_dict = MATPP_MODEL_ARCH
    head_layers = ['generator']
