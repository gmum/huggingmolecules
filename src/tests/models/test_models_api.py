import unittest

from torch import nn

from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from huggingmolecules.models.models_api import PretrainedModelBase
from tests.models.models_base import ModelsApiTestBase


class MockedConfig(PretrainedConfigMixin):
    d_model = 10

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        pass

    @classmethod
    def from_pretrained(cls, pretrained_name: str, **kwargs):
        return cls()


MOCKED_MODEL_ARCH = {}


class MockedModel(PretrainedModelBase):

    def __init__(self, config: MockedConfig):
        super().__init__(config)
        self.encoder = nn.Linear(10, config.d_model)
        self.decoder = nn.Linear(config.d_model, 10)
        self.generator = nn.Linear(10, 1)

    def forward(self, batch):
        return self.generator(self.decoder(self.encoder))

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        return MOCKED_MODEL_ARCH.get(pretrained_name, None)

    @classmethod
    def get_config_cls(cls):
        return MockedConfig

    @classmethod
    def get_featurizer_cls(cls):
        pass


class ModelsApiTest(ModelsApiTestBase, unittest.TestCase):
    config_cls = MockedConfig
    model_cls = MockedModel
    head_layers = ['generator']