from typing import Type, List, Optional, Generic, TypeVar

import torch
import torch.nn as nn

from experiments.wrappers import ChempropModelWrapper
from experiments.wrappers.wrappers_chemprop import ChempropFeaturizer
from src.huggingmolecules.featurization.featurization_api import BatchEncodingProtocol
from src.huggingmolecules.models.models_api import PretrainedModelBase


class EnsembleElement:
    def __init__(self, module_cls: Type[PretrainedModelBase], pretrained_name: str, weights_path: str):
        self.module_cls = module_cls
        self.pretrained_name = pretrained_name
        self.weights_path = weights_path
        self.model = None

    def attach(self, device):
        self.model = self.module_cls.from_pretrained(self.pretrained_name)
        self.model.load_weights(self.weights_path)
        self.model.to(device)

    def detach(self):
        del self.model
        torch.cuda.empty_cache()

    def __call__(self, batch: BatchEncodingProtocol):
        return self.model(batch)


class EnsembleModule(nn.Module):
    def __init__(self, models: Optional[List[EnsembleElement]] = None):
        super().__init__()
        self.models = models
        self.device = torch.device('cpu')

    def forward(self, batch: BatchEncodingProtocol):
        outputs = []
        for model in self.models:
            model.attach(self.device)
            outputs.append(model(batch))
            model.detach()

        return torch.mean(torch.stack(outputs))

