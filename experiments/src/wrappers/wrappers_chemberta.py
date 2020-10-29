import os
from dataclasses import dataclass
from typing import Tuple, List, Optional, Type

import torch

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from src.huggingmolecules.featurization.featurization_common_utils import stack_y_list
from src.huggingmolecules.models.models_api import PretrainedModelBase

try:
    import transformers
except ImportError:
    raise ImportError('Please install transformers (pip install transformers) '
                      'from https://github.com/huggingface/transformers to use ChembertaModelWrapper.')


@dataclass
class ChembertaConfig(PretrainedConfigMixin):
    pretrained_name: str = None

    @classmethod
    def from_pretrained(cls, pretrained_name: str, **kwargs):
        return cls(pretrained_name=pretrained_name)


@dataclass
class ChembertaBatchEncoding(RecursiveToDeviceMixin):
    data: dict
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class ChembertaFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], ChembertaBatchEncoding, ChembertaConfig]):
    def __init__(self, config: ChembertaConfig):
        super().__init__(config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.pretrained_name)

    def _collate_encodings(self, encodings: List[Tuple[dict, float]]) -> ChembertaBatchEncoding:
        x_list, y_list = zip(*encodings)
        padded = self.tokenizer.pad(x_list, return_tensors='pt')
        data = {k: v for k, v in padded.items()}
        return ChembertaBatchEncoding(data=data,
                                      y=stack_y_list(y_list),
                                      batch_size=len(x_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[dict, float]:
        return self.tokenizer.encode_plus(smiles), y


class ChembertaModelWrapper(PretrainedModelBase):
    def __init__(self, model, config):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        super().__init__(config)
        self.model = model

    @classmethod
    def get_featurizer_cls(cls) -> Type[ChembertaFeaturizer]:
        return ChembertaFeaturizer

    @classmethod
    def get_config_cls(cls) -> Type[ChembertaConfig]:
        return ChembertaConfig

    def forward(self, batch):
        return self.model(**batch.data)[0]

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name: str, *,
                        excluded: List[str] = None,
                        config: ChembertaConfig = None) -> "ChembertaModelWrapper":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_name, num_labels=1)
        if not config:
            config = ChembertaConfig()
        return cls(model, config)
