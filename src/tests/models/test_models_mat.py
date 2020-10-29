import unittest

from huggingmolecules import MatConfig, MatModel, MatFeaturizer
from tests.models.models_base import ModelsApiTestBase, ModelsForwardTestBase


class ModelsMatApiTest(ModelsApiTestBase, unittest.TestCase):
    config_cls = MatConfig
    model_cls = MatModel
    head_layers = ['generator']


class ModelsMatForwardTest(ModelsForwardTestBase, unittest.TestCase):
    model_cls = MatModel
    featurizer_cls = MatFeaturizer
    config_cls = MatConfig
