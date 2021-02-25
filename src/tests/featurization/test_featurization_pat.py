import unittest

from huggingmolecules import PatConfig, PatFeaturizer
from huggingmolecules.configuration.configuration_pat import PAT_CONFIG_ARCH
from tests.featurization.data.featurization_expected_pat import expected_batch, expected_encoded_smiles
from tests.featurization.featurization_base import FeaturizationApiTestBase


class FeaturizationApiPatTest(FeaturizationApiTestBase, unittest.TestCase):
    config_cls = PatConfig
    featurizer_cls = PatFeaturizer
    config_arch_dict = PAT_CONFIG_ARCH
    expected_encoded_smiles = {pretrained: expected_encoded_smiles for pretrained in config_arch_dict.keys()}
    expected_batch = {pretrained: expected_batch for pretrained in config_arch_dict.keys()}
