import unittest

from huggingmolecules import GroverConfig, GroverModel
from huggingmolecules.configuration.configuration_grover import GROVER_CONFIG_ARCH
from huggingmolecules.models.models_grover import GROVER_MODEL_ARCH
from tests.downloading.downloading_base import ConfigurationArchTestBase, ModelsArchTestBase


class ConfigurationGroverArchTest(ConfigurationArchTestBase, unittest.TestCase):
    config_cls = GroverConfig
    config_arch_dict = GROVER_CONFIG_ARCH


class ModelsGroverArchTest(unittest.TestCase, ModelsArchTestBase):
    model_cls = GroverModel
    model_arch_dict = GROVER_MODEL_ARCH
    head_layers = ['readout', 'mol_atom_from_atom_ffn', 'mol_atom_from_bond_ffn']
