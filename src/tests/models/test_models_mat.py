import unittest

from huggingmolecules import MatConfig, MatModel, MatFeaturizer
from huggingmolecules.models.models_mat import MAT_MODEL_ARCH
from tests.models.models_base import ModelsApiTestBase, ModelsArchTestBase, ModelsForwardTestBase


class ModelsMatApiTest(ModelsApiTestBase, unittest.TestCase):
    config_cls = MatConfig
    model_cls = MatModel
    head_layers = ['generator']


class ModelsMatArchTest(ModelsArchTestBase, unittest.TestCase):
    model_cls = MatModel
    model_arch_dict = MAT_MODEL_ARCH
    head_layers = ['generator']


class ModelsMatForwardTest(ModelsForwardTestBase, unittest.TestCase):
    model_cls = MatModel
    featurizer_cls = MatFeaturizer
    config_cls = MatConfig
