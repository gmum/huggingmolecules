from typing import Tuple

import numpy as np
from rdkit.Chem import Mol, Bond

from .featurization_common_utils import one_hot_vector


def get_bond_features(bond: Bond) -> np.ndarray:
    attributes = []
    attributes += one_hot_vector(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0])
    attributes.append(bond.GetIsAromatic())
    attributes.append(bond.GetIsConjugated())
    attributes.append(bond.IsInRing())

    return np.array(attributes, dtype=np.float32)


def floyd_warshall(adj: np.ndarray, inf: float = 999.0) -> np.ndarray:
    n_nodes = adj.shape[0]
    dist = inf * np.ones((n_nodes, n_nodes))
    dist[adj == 1] = 1
    np.fill_diagonal(dist, 0)

    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def get_relational_array(order_matrix: np.ndarray, max_order: int = 4, inf: float = 999.0) -> np.ndarray:
    relational_mat = np.zeros((max_order + 2, *order_matrix.shape))

    for i in range(max_order):
        relational_mat[i][order_matrix == i] = 1

    relational_mat[max_order][(order_matrix >= max_order) & (order_matrix < inf)] = 1
    relational_mat[max_order + 1][order_matrix == inf] = 1  # additional dim for dummy node

    return relational_mat


def build_bond_features_matrix(molecule: Mol) -> np.ndarray:
    bond_matrix = np.zeros((7, molecule.GetNumAtoms(), molecule.GetNumAtoms()))

    for bond in molecule.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        bond_features = get_bond_features(bond)
        bond_matrix[:, begin_atom, end_atom] = bond_features
        bond_matrix[:, end_atom, begin_atom] = bond_features

    return bond_matrix


def build_relative_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    order_matrix = floyd_warshall(adj_matrix)
    relative_mat = get_relational_array(order_matrix, max_order=4)

    return relative_mat


def add_mask_feature(bond_features: np.ndarray, node_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.zeros((node_features.shape[0], node_features.shape[1] + 1))
    m[:, 1:] = node_features
    node_features = m

    m = np.zeros((bond_features.shape[0] + 1, bond_features.shape[1], bond_features.shape[2]))
    m[1:, :, :] = bond_features
    bond_features = m

    return bond_features, node_features
