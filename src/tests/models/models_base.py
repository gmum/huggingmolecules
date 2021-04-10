import os
import random
import tempfile
from typing import Type, List

from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from huggingmolecules.models.models_api import PretrainedModelBase
from tests.common.api import AbstractTestCase
from tests.common.utils import assert_dicts_almost_equals, assert_negate, get_excluded_params


class ModelsApiTestBase(AbstractTestCase):
    config_cls: Type[PretrainedConfigMixin]
    model_cls: Type[PretrainedModelBase]
    head_layers: List[str]

    def setUp(self):
        super().setUp()
        self.config = self.config_cls()
        self.model = self.model_cls(self.config)
        self.excluded_params = get_excluded_params(self.model, self.head_layers)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            weight_file_path = os.path.join(tmp, 'weight.pt')
            self.model.save_weights(weight_file_path)
            assert os.path.exists(weight_file_path)
            model_second = self.model_cls(self.config)
            model_second.load_weights(weight_file_path)

        assert_dicts_almost_equals(self.model.state_dict(), model_second.state_dict())

    def test_save_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            weight_file_path = os.path.join(tmp, 'weight.pt')
            self.model.save_weights(weight_file_path, excluded=self.head_layers)
            model_second = self.model_cls(self.config)
            model_second.load_weights(weight_file_path)

        excluded = self.excluded_params
        assert_dicts_almost_equals(self.model.state_dict(), model_second.state_dict(), excluded=excluded)
        assert_negate(lambda: assert_dicts_almost_equals(self.model.state_dict(), model_second.state_dict()))

    def test_load_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            weight_file_path = os.path.join(tmp, 'weight.pt')
            self.model.save_weights(weight_file_path)
            model_second = self.model_cls(self.config)
            model_second.load_weights(weight_file_path, excluded=self.head_layers)

        excluded = self.excluded_params
        assert_dicts_almost_equals(self.model.state_dict(), model_second.state_dict(), excluded=excluded)
        assert_negate(lambda: assert_dicts_almost_equals(self.model.state_dict(), model_second.state_dict()))

    def test_from_pretrained_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            weight_file_path = os.path.join(tmp, 'weight.pt')
            self.model.save_weights(weight_file_path)
            model_second = self.model_cls.from_pretrained(weight_file_path, config=self.config)

        assert_dicts_almost_equals(self.model.state_dict(), model_second.state_dict())

    def test_from_pretrained_custom_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            weight_file_path = os.path.join(tmp, 'weight.pt')
            self.model.save_weights(weight_file_path)
            model_second = self.model_cls.from_pretrained(weight_file_path,
                                                          config=self.config,
                                                          excluded=self.head_layers)

        excluded = self.excluded_params
        assert_dicts_almost_equals(self.model.state_dict(), model_second.state_dict(), excluded=excluded)
        assert_negate(lambda: assert_dicts_almost_equals(self.model.state_dict(), model_second.state_dict()))


class ModelsForwardTestBase(AbstractTestCase):
    model_cls: Type[PretrainedModelBase]
    featurizer_cls: Type[PretrainedFeaturizerMixin]
    config_cls: Type[PretrainedConfigMixin]
    smiles_list = ['C/C=C/C', '[C]=O', 'CC(=O)O', 'C1CC1']

    def setUp(self):
        super().setUp()
        self.config = self.config_cls()
        self.featurizer = self.featurizer_cls(self.config)
        self.model = self.model_cls(self.config)

    def test_forward_pass(self):
        for _ in range(3):
            size = random.randint(1, 7)
            smiles = [random.choice(self.smiles_list) for _ in range(size)]
            batch = self.featurizer(smiles)
            output = self.model(batch)
            self.test.assertTupleEqual(output.shape, (size, 1))
