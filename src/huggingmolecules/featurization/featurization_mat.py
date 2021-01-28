from dataclasses import dataclass
from typing import *

import torch
from sklearn.metrics import pairwise_distances

from src.huggingmolecules import MatConfig
from .featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from .featurization_mat_utils import add_dummy_node, build_position_matrix, build_atom_features_matrix, \
    get_mol_from_smiles, build_adjacency_matrix, pad_sequence


@dataclass
class MatMoleculeEncoding:
    node_features: torch.FloatTensor
    adjacency_matrix: torch.FloatTensor
    distance_matrix: torch.FloatTensor
    y: Optional[torch.FloatTensor]


@dataclass
class MatBatchEncoding(RecursiveToDeviceMixin):
    node_features: torch.FloatTensor
    adjacency_matrix: torch.FloatTensor
    distance_matrix: torch.FloatTensor
    y: Optional[torch.FloatTensor]
    batch_size: int

    def __len__(self):
        return self.batch_size


class MatFeaturizer(PretrainedFeaturizerMixin[MatMoleculeEncoding, MatBatchEncoding, MatConfig]):
    @classmethod
    def get_config_cls(cls):
        return MatConfig

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> MatMoleculeEncoding:
        mol = get_mol_from_smiles(smiles)

        node_features = build_atom_features_matrix(mol)
        adj_matrix = build_adjacency_matrix(mol)
        pos_matrix = build_position_matrix(mol)
        dist_matrix = pairwise_distances(pos_matrix)

        node_features, adj_matrix, dist_matrix, _ = add_dummy_node(node_features=node_features,
                                                                   adj_matrix=adj_matrix,
                                                                   dist_matrix=dist_matrix)

        return MatMoleculeEncoding(node_features=node_features,
                                   adjacency_matrix=adj_matrix,
                                   distance_matrix=dist_matrix,
                                   y=y)

    def _collate_encodings(self, encodings: List[MatMoleculeEncoding]) -> MatBatchEncoding:
        node_features = pad_sequence([torch.tensor(e.node_features).float() for e in encodings])
        adj_matrix = pad_sequence([torch.tensor(e.adjacency_matrix).float() for e in encodings])
        dist_matrix = pad_sequence([torch.tensor(e.distance_matrix).float() for e in encodings])
        y = None if any(e.y is None for e in encodings) \
            else torch.stack([torch.tensor(e.y).float() for e in encodings]).unsqueeze(1)

        return MatBatchEncoding(node_features=node_features,
                                adjacency_matrix=adj_matrix,
                                distance_matrix=dist_matrix,
                                y=y,
                                batch_size=len(encodings))
