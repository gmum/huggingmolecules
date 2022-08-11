import unittest

from huggingmolecules import RMatConfig, RMatModel
from huggingmolecules.configuration.configuration_rmat import RMAT_CONFIG_ARCH
from huggingmolecules.models.models_rmat import RMAT_MODEL_ARCH
from tests.downloading.downloading_base import ConfigurationArchTestBase, ModelsArchTestBase


class ConfigurationRMatArchTest(ConfigurationArchTestBase, unittest.TestCase):
    config_cls = RMatConfig
    config_arch_dict = RMAT_CONFIG_ARCH


class ModelsRMatArchTest(ModelsArchTestBase, unittest.TestCase):
    model_cls = RMatModel
    model_arch_dict = RMAT_MODEL_ARCH
    head_layers = ['generator']
