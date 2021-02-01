from dataclasses import dataclass
from typing import *

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from src.huggingmolecules import PatConfig
from .featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from .featurization_common_utils import stack_y
from .featurization_mat_utils import add_dummy_node, build_position_matrix, build_atom_features_matrix, \
    get_mol_from_smiles, build_adjacency_matrix, pad_sequence
from .featurization_pat_utils import build_bond_features_matrix, \
    build_relative_matrix, add_mask_feature


@dataclass
class PatMoleculeEncoding:
    node_features: np.ndarray
    bond_features: np.ndarray
    distance_matrix: np.ndarray
    relative_matrix: np.ndarray
    y: Optional[np.ndarray]


@dataclass
class PatBatchEncoding(RecursiveToDeviceMixin):
    node_features: torch.FloatTensor
    bond_features: torch.FloatTensor
    relative_matrix: torch.FloatTensor
    distance_matrix: torch.FloatTensor
    y: Optional[torch.FloatTensor]
    batch_size: int

    def __len__(self):
        return self.batch_size


class PatFeaturizer(PretrainedFeaturizerMixin[PatMoleculeEncoding, PatBatchEncoding, PatConfig]):
    @classmethod
    def get_config_cls(cls):
        return PatConfig

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> PatMoleculeEncoding:
        mol = get_mol_from_smiles(smiles)

        node_features = build_atom_features_matrix(mol)
        bond_features = build_bond_features_matrix(mol)
        adj_matrix = build_adjacency_matrix(mol)
        pos_matrix = build_position_matrix(mol)
        dist_matrix = pairwise_distances(pos_matrix)

        node_features, adj_matrix, dist_matrix, bond_features = add_dummy_node(node_features=node_features,
                                                                               adj_matrix=adj_matrix,
                                                                               dist_matrix=dist_matrix,
                                                                               bond_features=bond_features)
        relative_matrix = build_relative_matrix(adj_matrix)

        bond_features, node_features = add_mask_feature(bond_features, node_features)

        return PatMoleculeEncoding(node_features=node_features,
                                   bond_features=bond_features,
                                   distance_matrix=dist_matrix,
                                   relative_matrix=relative_matrix,
                                   y=y)

    def _collate_encodings(self, encodings: List[PatMoleculeEncoding]) -> PatBatchEncoding:
        node_features = pad_sequence([torch.tensor(e.node_features).float() for e in encodings])
        dist_matrix = pad_sequence([torch.tensor(e.distance_matrix).float() for e in encodings])
        bond_features = pad_sequence([torch.tensor(e.bond_features).float() for e in encodings])
        relative_matrix = pad_sequence([torch.tensor(e.relative_matrix).float() for e in encodings])

        return PatBatchEncoding(node_features=node_features,
                                bond_features=bond_features,
                                relative_matrix=relative_matrix,
                                distance_matrix=dist_matrix,
                                y=stack_y(encodings),
                                batch_size=len(encodings))
