from dataclasses import dataclass
from typing import *

import numpy as np
import torch
from chemprop.data import MoleculeDatapoint

from src.huggingmolecules import GroverConfig
from .featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from .featurization_grover_utils import BatchMolGraph, MolGraph


@dataclass
class GroverMoleculeEncoding:
    mol_graph: "MolGraph"
    features: np.array
    y: Optional[float]


@dataclass
class GroverBatchEncoding(RecursiveToDeviceMixin):
    batch_mol_graph: "BatchMolGraph"
    batch_features: Optional[List[torch.Tensor]]
    y: Optional[torch.FloatTensor]
    batch_size: int

    def __len__(self):
        return self.batch_size


class GroverFeaturizer(PretrainedFeaturizerMixin[GroverMoleculeEncoding, GroverBatchEncoding, GroverConfig]):
    @classmethod
    def get_config_cls(cls):
        return GroverConfig

    def __init__(self, config: GroverConfig, features_generator: Optional[List[str]] = None):
        super().__init__(config)
        self.features_generator = features_generator

    def _collate_encodings(self, encodings: List[GroverMoleculeEncoding]) -> GroverBatchEncoding:
        batch_mol_graph = BatchMolGraph([e.mol_graph for e in encodings])
        batch_features = [torch.tensor(e.features).float() for e in encodings] \
            if encodings[0].features is not None else None
        y = torch.stack([torch.tensor(e.y).float() for e in encodings]).view(-1, 1) \
            if encodings[0].y is not None else None

        return GroverBatchEncoding(batch_mol_graph=batch_mol_graph,
                                   batch_features=batch_features,
                                   y=y,
                                   batch_size=len(encodings))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> GroverMoleculeEncoding:
        datapoint = MoleculeDatapoint([smiles], features_generator=self.features_generator)
        mol_graph = MolGraph(datapoint.mol[0])
        features = datapoint.features
        return GroverMoleculeEncoding(mol_graph, features, y)
