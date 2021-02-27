import unittest

from huggingmolecules import MatConfig, MatModel
from huggingmolecules.configuration.configuration_mat import MAT_CONFIG_ARCH
from huggingmolecules.models.models_mat import MAT_MODEL_ARCH
from tests.downloading.downloading_base import ConfigurationArchTestBase, ModelsArchTestBase


class ConfigurationMatArchTest(ConfigurationArchTestBase, unittest.TestCase):
    config_cls = MatConfig
    config_arch_dict = MAT_CONFIG_ARCH


class ModelsMatArchTest(ModelsArchTestBase, unittest.TestCase):
    model_cls = MatModel
    model_arch_dict = MAT_MODEL_ARCH
    head_layers = ['generator']
