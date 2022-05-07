import unittest

from huggingmolecules import RMatConfig, RMatModel, RMatFeaturizer
from tests.models.models_base import ModelsApiTestBase, ModelsForwardTestBase


class ModelsRMatApiTest(ModelsApiTestBase, unittest.TestCase):
    config_cls = RMatConfig
    model_cls = RMatModel
    head_layers = ['generator']


class ModelsRMatForwardTest(ModelsForwardTestBase, unittest.TestCase):
    model_cls = RMatModel
    featurizer_cls = RMatFeaturizer
    config_cls = RMatConfig
