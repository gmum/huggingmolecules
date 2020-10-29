"""
This implementation is copied from
https://github.com/ardigen/MAT/tree/master/src/featurization/data_utils.py
"""

from typing import Tuple, List, TypeVar

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmiles

from ..featurization.featurization_common_utils import one_hot_vector

T_Tensor = TypeVar('T_Tensor', bound=torch.Tensor)


def pad_array(array: T_Tensor, *, size: Tuple[int, ...], dtype: torch.dtype = None) -> T_Tensor:
    if dtype is None:
        dtype = array.dtype
    result = torch.zeros(size=size, dtype=dtype)
    slices = tuple(slice(s) for s in array.shape)
    result[slices] = array
    return result


def pad_sequence(sequence: List[T_Tensor], dtype: torch.dtype = None) -> T_Tensor:
    shapes = torch.stack([torch.tensor(t.shape) for t in sequence])
    max_shape = tuple(torch.max(shapes, dim=0).values)
    return torch.stack([pad_array(t, size=max_shape, dtype=dtype) for t in sequence])


def add_dummy_node(*, node_features: np.ndarray = None,
                   adj_matrix: np.ndarray = None,
                   dist_matrix: np.ndarray = None,
                   bond_features: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if node_features is not None:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.0
        node_features = m
    if adj_matrix is not None:
        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m
    if dist_matrix is not None:
        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m
    if bond_features is not None:
        m = np.zeros((bond_features.shape[0], bond_features.shape[1] + 1, bond_features.shape[2] + 1))
        m[:, 1:, 1:] = bond_features
        bond_features = m

    return node_features, adj_matrix, dist_matrix, bond_features


def build_position_matrix(molecule: Mol) -> np.ndarray:
    conf = molecule.GetConformer()
    return np.array(
        [
            [
                conf.GetAtomPosition(k).x,
                conf.GetAtomPosition(k).y,
                conf.GetAtomPosition(k).z,
            ]
            for k in range(molecule.GetNumAtoms())
        ]
    )


def build_atom_features_matrix(mol: Mol) -> np.ndarray:
    return np.array([get_atom_features(atom) for atom in mol.GetAtoms()])


def get_atom_features(atom) -> np.ndarray:
    features = []

    features += one_hot_vector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    features += one_hot_vector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    features += one_hot_vector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features += one_hot_vector(atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    features.append(atom.IsInRing())
    features.append(atom.GetIsAromatic())

    return np.array(features, dtype=np.float32)


def get_mol_from_smiles(smiles: str) -> Mol:
    mol = MolFromSmiles(smiles)
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, maxAttempts=5000)
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
    except ValueError:
        mol = MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)

    return mol


def build_adjacency_matrix(molecule: Mol) -> np.ndarray:
    adj_matrix = np.eye(molecule.GetNumAtoms())

    for bond in molecule.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    return adj_matrix
