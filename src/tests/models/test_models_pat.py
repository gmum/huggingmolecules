import unittest

from huggingmolecules import PatConfig, PatModel, PatFeaturizer
from tests.models.models_base import ModelsApiTestBase, ModelsForwardTestBase


class ModelsPatApiTest(ModelsApiTestBase, unittest.TestCase):
    config_cls = PatConfig
    model_cls = PatModel
    head_layers = ['generator']


class ModelsPatForwardTest(ModelsForwardTestBase, unittest.TestCase):
    model_cls = PatModel
    featurizer_cls = PatFeaturizer
    config_cls = PatConfig
