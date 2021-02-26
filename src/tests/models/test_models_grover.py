import unittest
import random

from huggingmolecules import GroverModel, GroverConfig, GroverFeaturizer
from huggingmolecules.models.models_grover import GROVER_MODEL_ARCH
from tests.models.models_base import ModelsApiTestBase, ModelsArchTestBase, ModelsForwardTestBase


class ModelsGroverApiTest(ModelsApiTestBase, unittest.TestCase):
    config_cls = GroverConfig
    model_cls = GroverModel
    head_layers = ['readout']


class ModelsGroverArchTest(unittest.TestCase, ModelsArchTestBase):
    model_cls = GroverModel
    model_arch_dict = GROVER_MODEL_ARCH
    head_layers = ['readout']


class ModelsGroverForwardTest(ModelsForwardTestBase, unittest.TestCase):
    model_cls = GroverModel
    featurizer_cls = GroverFeaturizer
    config_cls = GroverConfig

    def test_forward_pass(self):
        for _ in range(3):
            size = random.randint(1, 7)
            smiles = [random.choice(self.smiles_list) for _ in range(size)]
            batch = self.featurizer(smiles)
            output = self.model(batch)
            self.test.assertEqual(len(output), 2)
            self.test.assertTupleEqual(output[0].shape, (size, 1))
            self.test.assertTupleEqual(output[1].shape, (size, 1))
