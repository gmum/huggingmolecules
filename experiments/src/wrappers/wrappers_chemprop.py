from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
from chemprop.args import TrainArgs
from chemprop.data import MoleculeDatapoint
from chemprop.features import MolGraph, BatchMolGraph
from chemprop.models import MoleculeModel

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase


@dataclass
class ChempropBatchEncoding(RecursiveToDeviceMixin):
    batch_mol_graph: BatchMolGraph
    batch_features: Optional[List[torch.Tensor]]
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class ChempropFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], ChempropBatchEncoding]):
    def __init__(self, features_generator: Optional[List[str]] = None):
        self.features_generator = features_generator

    def _collate_encodings(self, encodings: List[Tuple[MolGraph, Optional[np.array], float]]) -> ChempropBatchEncoding:
        batch_mol_graph = BatchMolGraph([e[0] for e in encodings])
        batch_features = [torch.tensor(e[1]).float() for e in encodings] if encodings[0][1] is not None else None
        y_list = [torch.tensor(e[2]).float() for e in encodings]
        return ChempropBatchEncoding(batch_mol_graph,
                                     batch_features,
                                     torch.stack(y_list).float().view(-1, 1),
                                     len(y_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[MolGraph, np.array, float]:
        datapoint = MoleculeDatapoint([smiles], features_generator=self.features_generator)
        mol_graph = MolGraph(datapoint.mol[0])
        features = datapoint.features
        return mol_graph, features, y

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        if pretrained_name == 'vanilla':
            return cls()
        else:
            return cls(pretrained_name.split("+"))


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
        return self.model([batch.batch_mol_graph], batch.batch_features)

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_name: str, task: str, **kwargs):
        args = TrainArgs()
        args.parse_args(args=["--data_path", "non_existent", "--dataset_type", task])
        args.task_names = ["whatever"]  # taks_num must be > 0
        model = MoleculeModel(args)
        return cls(model)
