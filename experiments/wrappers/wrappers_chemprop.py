from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch_geometric

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
import chemprop

from src.huggingmolecules.models.models_api import PretrainedModelBase


@dataclass
class ChempropBatchEncoding(torch_geometric.data.Data):
    batch_mol_graph: chemprop.features.BatchMolGraph
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class ChempropFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], ChempropBatchEncoding]):
    def _collate_encodings(self, encodings: List[Tuple[chemprop.features.MolGraph, float]]) -> ChempropBatchEncoding:
        batch_mol_graph = chemprop.features.BatchMolGraph([e[0] for e in encodings])
        y_list = [torch.tensor(e[1]).float() for e in encodings]
        return ChempropBatchEncoding(batch_mol_graph, torch.stack(y_list).float().view(-1, 1), len(y_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[chemprop.features.MolGraph, float]:
        mol_graph = chemprop.features.MolGraph(smiles)
        return mol_graph, y

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        if pretrained_name == 'vanilla':
            return cls()
        else:
            raise NotImplementedError


@dataclass
class ChempropConfig(PretrainedConfigMixin):
    d_model: int = 1  # for better NoamLRScheduler control


class ChempropModelWrapper(PretrainedModelBase):
    def __init__(self, model):
        super().__init__(ChempropConfig())
        self.model = model

    @classmethod
    def get_featurizer_cls(cls):
        return ChempropFeaturizer

    @classmethod
    def get_config_cls(cls):
        return ChempropConfig

    def forward(self, batch: ChempropBatchEncoding):
        return self.model([batch.batch_mol_graph])

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        if pretrained_name == 'vanilla':
            args = chemprop.args.TrainArgs()
            args.parse_args(args=["--data_path", "non_existent", "--dataset_type", "regression"])
            args.task_names = ["whatever"]
            model = chemprop.models.MoleculeModel(args)
            return cls(model)
        else:
            raise NotImplementedError
