import unittest

from huggingmolecules import GroverConfig, GroverFeaturizer
from huggingmolecules.configuration.configuration_grover import GROVER_CONFIG_ARCH
from tests.featurization.expected.featurization_expected_grover import expected_batch, expected_encoded_smiles
from tests.featurization.featurization_base import FeaturizationApiTestBase


class FeaturizationApiGroverTest(FeaturizationApiTestBase, unittest.TestCase):
    config_cls = GroverConfig
    featurizer_cls = GroverFeaturizer
    config_arch_dict = GROVER_CONFIG_ARCH
    expected_encoded_smiles = {pretrained: expected_encoded_smiles for pretrained in config_arch_dict.keys()}
    expected_batch = {pretrained: expected_batch for pretrained in config_arch_dict.keys()}
