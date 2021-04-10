import unittest

from huggingmolecules import MatppConfig, MatppModel, MatppFeaturizer
from tests.models.models_base import ModelsApiTestBase, ModelsForwardTestBase


class ModelsMatppApiTest(ModelsApiTestBase, unittest.TestCase):
    config_cls = MatppConfig
    model_cls = MatppModel
    head_layers = ['generator']


class ModelsMatppForwardTest(ModelsForwardTestBase, unittest.TestCase):
    model_cls = MatppModel
    featurizer_cls = MatppFeaturizer
    config_cls = MatppConfig
