import unittest

from huggingmolecules import RMatConfig, RMatFeaturizer
from huggingmolecules.configuration.configuration_rmat import RMAT_CONFIG_ARCH
from tests.featurization.expected.featurization_expected_rmat import expected_batch, expected_encoded_smiles
from tests.featurization.featurization_base import FeaturizationApiTestBase


class FeaturizationApiRMatTest(FeaturizationApiTestBase, unittest.TestCase):
    config_cls = RMatConfig
    featurizer_cls = RMatFeaturizer
    config_arch_dict = RMAT_CONFIG_ARCH
    expected_encoded_smiles = {pretrained: expected_encoded_smiles for pretrained in config_arch_dict.keys()}
    expected_batch = {pretrained: expected_batch for pretrained in config_arch_dict.keys()}
