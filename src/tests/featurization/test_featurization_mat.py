import unittest

from huggingmolecules import MatConfig, MatFeaturizer
from huggingmolecules.configuration.configuration_mat import MAT_CONFIG_ARCH
from tests.featurization.expected.featurization_expected_mat import expected_batch, expected_encoded_smiles
from tests.featurization.featurization_base import FeaturizationApiTestBase


class FeaturizationApiMatTest(FeaturizationApiTestBase, unittest.TestCase):
    config_cls = MatConfig
    featurizer_cls = MatFeaturizer
    config_arch_dict = MAT_CONFIG_ARCH
    expected_encoded_smiles = {pretrained: expected_encoded_smiles for pretrained in config_arch_dict.keys()}
    expected_batch = {pretrained: expected_batch for pretrained in config_arch_dict.keys()}
