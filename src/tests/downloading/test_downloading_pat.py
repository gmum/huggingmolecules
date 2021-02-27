import unittest

from huggingmolecules import PatConfig, PatModel
from huggingmolecules.configuration.configuration_pat import PAT_CONFIG_ARCH
from huggingmolecules.models.models_pat import PAT_MODEL_ARCH
from tests.downloading.downloading_base import ConfigurationArchTestBase, ModelsArchTestBase


class ConfigurationPatArchTest(ConfigurationArchTestBase, unittest.TestCase):
    config_cls = PatConfig
    config_arch_dict = PAT_CONFIG_ARCH


class ModelsPatArchTest(ModelsArchTestBase, unittest.TestCase):
    model_cls = PatModel
    model_arch_dict = PAT_MODEL_ARCH
    head_layers = ['generator']
