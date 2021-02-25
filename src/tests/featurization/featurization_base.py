import os
import random
import tempfile
from typing import Type, List

import numpy as np
import torch

from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin


class FeaturizationApiTestBase:
    config_cls: Type[PretrainedConfigMixin]
    featurizer_cls: Type[PretrainedFeaturizerMixin]
    config_arch_dict: dict
    smiles_list: List[str] = ['C/C=C/C', '[C]=O']
    expected_encoded_smiles: dict
    expected_batch: dict

    def assert_arrays_equals(self, a, b):
        if a is None or b is None:
            assert a == b
        else:
            if torch.is_tensor(a) or torch.is_tensor(b):
                torch.testing.assert_allclose(a, b)
            elif isinstance(a, list) or isinstance(b, list):
                assert a == b
            else:
                assert np.allclose(a, b)

    def assert_encoding_equals(self, a, b, excluded=None):
        excluded = excluded if excluded else []
        assert a.__dict__.keys() == b.__dict__.keys()
        for key in a.__dict__.keys():
            if key not in excluded:
                self.assert_arrays_equals(a.__dict__[key], b.__dict__[key])

    def get_random_arch_key(self):
        return random.choice(list(self.config_arch_dict.keys()))

    def test_constructor_simple(self):
        config = self.config_cls.from_pretrained(self.get_random_arch_key())
        featurizer = self.featurizer_cls(config)

    def test_from_pretrained_simple(self):
        featurizer = self.featurizer_cls.from_pretrained(self.get_random_arch_key())

    def test_from_pretrained_custom_config(self):
        config = self.config_cls.from_pretrained(self.get_random_arch_key())
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, 'config.json')
            config.save(json_path)
            featurizer = self.featurizer_cls.from_pretrained(json_path)

        assert config.to_dict() == featurizer.config.to_dict()

    def test_encoding_attributes(self):
        for pretrained_name in self.config_arch_dict.keys():
            featurizer = self.featurizer_cls.from_pretrained(pretrained_name)
            batch = featurizer(self.smiles_list)
            assert hasattr(batch, 'y')
            assert hasattr(batch, '__len__')

    def test_encode_smiles_list(self):
        for pretrained_name in self.config_arch_dict.keys():
            featurizer = self.featurizer_cls.from_pretrained(pretrained_name)
            encoded = featurizer.encode_smiles_list(self.smiles_list)
            for res, exp in zip(encoded, self.expected_encoded_smiles[pretrained_name]):
                self.assert_encoding_equals(res, exp)
                assert res.y is None

    def test_encode_batch(self):
        for pretrained_name in self.config_arch_dict.keys():
            featurizer = self.featurizer_cls.from_pretrained(pretrained_name)
            batch = featurizer(self.smiles_list)
            self.assert_encoding_equals(batch, self.expected_batch[pretrained_name])
            assert batch.y is None

    def test_encode_smiles_y(self):
        y_list = [random.random() for _ in range(len(self.smiles_list))]
        for pretrained_name in self.config_arch_dict.keys():
            featurizer = self.featurizer_cls.from_pretrained(pretrained_name)
            encoded = featurizer.encode_smiles_list(self.smiles_list, y_list)
            for res, exp, y in zip(encoded, self.expected_encoded_smiles[pretrained_name], y_list):
                self.assert_encoding_equals(res, exp, excluded=['y'])
                assert res.y == y

    def test_encode_batch_y(self):
        y_list = [random.random() for _ in range(len(self.smiles_list))]
        for pretrained_name in self.config_arch_dict.keys():
            featurizer = self.featurizer_cls.from_pretrained(pretrained_name)
            batch = featurizer(self.smiles_list, y_list)
            self.assert_encoding_equals(batch, self.expected_batch[pretrained_name], excluded=['y'])
            self.assert_arrays_equals(batch.y, np.expand_dims(np.array(y_list), 1))

    def test_batch_size(self):
        smiles_list = [random.choice(self.smiles_list) for _ in range(random.randint(1, 5))]
        for pretrained_name in self.config_arch_dict.keys():
            featurizer = self.featurizer_cls.from_pretrained(pretrained_name)
            batch = featurizer(smiles_list)
            assert len(batch) == len(smiles_list)

    def test_y_size(self):
        smiles_list = [random.choice(self.smiles_list) for _ in range(random.randint(1, 5))]
        y_list = [random.random() for _ in range(len(smiles_list))]
        for pretrained_name in self.config_arch_dict.keys():
            featurizer = self.featurizer_cls.from_pretrained(pretrained_name)
            batch = featurizer(smiles_list, y_list)
            assert batch.y.shape == (len(y_list), 1)
