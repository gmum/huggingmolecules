import logging
from dataclasses import dataclass
from typing import *

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances

from .featurization_api import PretrainedFeaturizerBase
from .featurization_utils import pad_array


@dataclass
class MatMoleculeEncoding:
    node_features: np.ndarray
    adjacency_matrix: np.ndarray
    distance_matrix: np.ndarray
    y: Optional[np.ndarray]


@dataclass
class MatBatchEncoding:
    node_features: torch.FloatTensor
    adjacency_matrix: torch.FloatTensor
    distance_matrix: torch.FloatTensor
    batch_mask: torch.BoolTensor
    y: Optional[torch.FloatTensor]


class MatFeaturizer(PretrainedFeaturizerBase[MatMoleculeEncoding, MatBatchEncoding]):

    def __init__(self, add_dummy_node: bool = True, one_hot_formal_charge: bool = True):
        self._add_dummy_node = add_dummy_node
        self._one_hot_formal_charge = one_hot_formal_charge

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> MatMoleculeEncoding:
        try:
            mol = Chem.MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(mol, self._add_dummy_node, self._one_hot_formal_charge)
            return MatMoleculeEncoding(node_features=afm, adjacency_matrix=adj, distance_matrix=dist, y=y)
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    def _get_batch_from_encodings(self, encodings: List[MatMoleculeEncoding]) -> MatBatchEncoding:
        adjacency_list, distance_list, features_list, batch_mask_list, y_list = [], [], [], [], []
        max_size = max(e.adjacency_matrix.shape[0] for e in encodings)
        for e in encodings:
            adjacency_list.append(pad_array(e.adjacency_matrix, shape=(max_size, max_size)))
            distance_list.append(pad_array(e.distance_matrix, shape=(max_size, max_size)))
            features_list.append(pad_array(e.node_features, shape=(max_size, e.node_features.shape[1])))
            batch_mask = np.sum(np.abs(e.node_features), axis=-1) != 0
            batch_mask_list.append(pad_array(batch_mask, shape=max_size, dtype=np.bool))
            y_list.append(e.y)

        return MatBatchEncoding(node_features=torch.FloatTensor(features_list),
                                adjacency_matrix=torch.FloatTensor(adjacency_list),
                                distance_matrix=torch.FloatTensor(distance_list),
                                batch_mask=torch.BoolTensor(batch_mask_list),
                                y=None if None in y_list else torch.FloatTensor(y_list))


def featurize_mol(mol: Chem.rdchem.Mol, add_dummy_node: bool, one_hot_formal_charge: bool) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom: Chem.rdchem.Atom, one_hot_formal_charge: bool = True) -> np.ndarray:
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val: int, lst: List[int]) -> Iterable:
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)
