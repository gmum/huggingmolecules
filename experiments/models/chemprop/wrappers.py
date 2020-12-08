from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch_geometric

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
import chemprop


@dataclass
class ChemPropBatchEncoding(torch_geometric.data.Data):
    batch_mol_graph: chemprop.features.BatchMolGraph
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class ChemPropFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], ChemPropBatchEncoding]):
    def _collate_encodings(self, encodings: List[Tuple[chemprop.features.MolGraph, float]]) -> ChemPropBatchEncoding:
        batch_mol_graph = chemprop.features.BatchMolGraph([e[0] for e in encodings])
        y_list = [torch.tensor(e[1]).float() for e in encodings]
        return ChemPropBatchEncoding(batch_mol_graph, torch.stack(y_list).float().view(-1, 1), len(y_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[chemprop.features.MolGraph, float]:
        mol_graph = chemprop.features.MolGraph(smiles)
        return mol_graph, y


class ChemPropModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch: ChemPropBatchEncoding):
        return self.model([batch.batch_mol_graph])

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    def get_config(self):
        return ChemPropConfig()


@dataclass
class ChemPropConfig(PretrainedConfigMixin):
    d_model: int = 1  # for better NoamLRScheduler control
