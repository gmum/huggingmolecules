import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase


@dataclass
class ChembertaBatchEncoding(RecursiveToDeviceMixin):
    data: dict
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class ChembertaFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], ChembertaBatchEncoding]):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _collate_encodings(self, encodings: List[Tuple[dict, float]]) -> ChembertaBatchEncoding:
        x_list = [e[0] for e in encodings]
        y_list = [torch.tensor(e[1]).float() for e in encodings]
        padded = self.tokenizer.pad(x_list, return_tensors='pt')
        padded = {k: v for k, v in padded.items()}
        return ChembertaBatchEncoding(padded, torch.stack(y_list).float().view(-1, 1), len(x_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[dict, float]:
        return self.tokenizer.encode_plus(smiles), y

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        return cls(tokenizer)


@dataclass
class ChembertaConfig(PretrainedConfigMixin):
    d_model: int = 1024


class ChembertaModelWrapper(PretrainedModelBase):
    def __init__(self, model):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        super().__init__(ChembertaConfig())
        self.model = model

    @classmethod
    def get_featurizer_cls(cls):
        return ChembertaFeaturizer

    @classmethod
    def get_config_cls(cls):
        return ChembertaConfig

    def forward(self, batch):
        return self.model(**batch.data)[0]

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_name: str, task: str, **kwargs):
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, num_labels=1)
        return cls(model)
