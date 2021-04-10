import unittest

from huggingmolecules import MatppConfig, MatppFeaturizer
from huggingmolecules.configuration.configuration_matpp import MATPP_CONFIG_ARCH
from tests.featurization.expected.featurization_expected_matpp import expected_batch, expected_encoded_smiles
from tests.featurization.featurization_base import FeaturizationApiTestBase


class FeaturizationApiMatppTest(FeaturizationApiTestBase, unittest.TestCase):
    config_cls = MatppConfig
    featurizer_cls = MatppFeaturizer
    config_arch_dict = MATPP_CONFIG_ARCH
    expected_encoded_smiles = {pretrained: expected_encoded_smiles for pretrained in config_arch_dict.keys()}
    expected_batch = {pretrained: expected_batch for pretrained in config_arch_dict.keys()}
