from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch_geometric

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin


@dataclass
class ChemBertBatchEncoding(torch_geometric.data.Data):
    data: dict
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class ChemBertFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], ChemBertBatchEncoding]):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _collate_encodings(self, encodings: List[Tuple[dict, float]]) -> ChemBertBatchEncoding:
        x_list = [e[0] for e in encodings]
        y_list = [torch.tensor(e[1]).float() for e in encodings]
        padded = self.tokenizer.pad(x_list, return_tensors='pt')
        padded = {k: v for k, v in padded.items()}
        return ChemBertBatchEncoding(padded, torch.stack(y_list).float().view(-1, 1), len(x_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[dict, float]:
        return self.tokenizer.encode_plus(smiles), y


class ChemBertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(**batch.data)[0]

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    def get_config(self):
        return ChemBertConfig()


@dataclass
class ChemBertConfig(PretrainedConfigMixin):
    d_model: int = 1024
