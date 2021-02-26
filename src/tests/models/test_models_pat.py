import unittest

from huggingmolecules import PatConfig, PatModel, PatFeaturizer
from huggingmolecules.models.models_pat import PAT_MODEL_ARCH
from tests.models.models_base import ModelsApiTestBase, ModelsArchTestBase, ModelsForwardTestBase


class ModelsPatApiTest(ModelsApiTestBase, unittest.TestCase):
    config_cls = PatConfig
    model_cls = PatModel
    head_layers = ['generator']


class ModelsPatArchTest(ModelsArchTestBase, unittest.TestCase):
    model_cls = PatModel
    model_arch_dict = PAT_MODEL_ARCH
    head_layers = ['generator']


class ModelsPatForwardTest(ModelsForwardTestBase, unittest.TestCase):
    model_cls = PatModel
    featurizer_cls = PatFeaturizer
    config_cls = PatConfig
