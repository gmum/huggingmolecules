from dataclasses import dataclass
from typing import *

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from .featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from .featurization_common_utils import stack_y, generate_additional_features, stack_generated_features
from .featurization_mat_utils import add_dummy_node, build_position_matrix, build_atom_features_matrix, \
    get_mol_from_smiles, build_adjacency_matrix, pad_sequence
from .featurization_rmat_utils import build_bond_features_matrix, \
    build_relative_matrix, add_mask_feature
from ..configuration import RMatConfig


@dataclass
class RMatMoleculeEncoding:
    node_features: np.ndarray
    bond_features: np.ndarray
    distance_matrix: np.ndarray
    relative_matrix: np.ndarray
    generated_features: Optional[List[float]]
    y: Optional[np.ndarray]


@dataclass
class RMatBatchEncoding(RecursiveToDeviceMixin):
    node_features: torch.FloatTensor
    bond_features: torch.FloatTensor
    relative_matrix: torch.FloatTensor
    distance_matrix: torch.FloatTensor
    generated_features: Optional[torch.FloatTensor]
    y: Optional[torch.FloatTensor]
    batch_size: int

    def __len__(self):
        return self.batch_size


class RMatFeaturizer(PretrainedFeaturizerMixin[RMatMoleculeEncoding, RMatBatchEncoding, RMatConfig]):
    @classmethod
    def _get_config_cls(cls) -> Type[RMatConfig]:
        return RMatConfig

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> RMatMoleculeEncoding:
        mol = get_mol_from_smiles(smiles)

        node_features = build_atom_features_matrix(mol)
        bond_features = build_bond_features_matrix(mol)
        adj_matrix = build_adjacency_matrix(mol)
        pos_matrix = build_position_matrix(mol)
        dist_matrix = pairwise_distances(pos_matrix)
        generated_features = generate_additional_features(mol, self.config.generator_features_generators)

        node_features, adj_matrix, dist_matrix, bond_features = add_dummy_node(node_features=node_features,
                                                                               adj_matrix=adj_matrix,
                                                                               dist_matrix=dist_matrix,
                                                                               bond_features=bond_features)
        relative_matrix = build_relative_matrix(adj_matrix)
        bond_features, node_features = add_mask_feature(bond_features, node_features)

        return RMatMoleculeEncoding(node_features=node_features,
                                    bond_features=bond_features,
                                    distance_matrix=dist_matrix,
                                    relative_matrix=relative_matrix,
                                    generated_features=generated_features,
                                    y=y)

    def _collate_encodings(self, encodings: List[RMatMoleculeEncoding]) -> RMatBatchEncoding:
        node_features = pad_sequence([torch.tensor(e.node_features).float() for e in encodings])
        dist_matrix = pad_sequence([torch.tensor(e.distance_matrix).float() for e in encodings])
        bond_features = pad_sequence([torch.tensor(e.bond_features).float() for e in encodings])
        relative_matrix = pad_sequence([torch.tensor(e.relative_matrix).float() for e in encodings])

        return RMatBatchEncoding(node_features=node_features,
                                 bond_features=bond_features,
                                 relative_matrix=relative_matrix,
                                 distance_matrix=dist_matrix,
                                 generated_features=stack_generated_features(encodings),
                                 y=stack_y(encodings),
                                 batch_size=len(encodings))
