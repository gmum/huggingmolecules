import logging
import numpy as np
import torch
from rdkit.Chem import MolFromSmiles

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor


class MatFeaturizer:

    def __init__(self, add_dummy_node=True, one_hot_formal_charge=True):
        self._add_dummy_node = add_dummy_node
        self._one_hot_formal_charge = one_hot_formal_charge

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, smiles_list, padding=False, return_tensors=None):
        adjacency_list, distance_list, features_list, mask_list = [], [], [], []
        for smiles in smiles_list:
            try:
                mol = MolFromSmiles(smiles)
                try:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, maxAttempts=5000)
                    AllChem.UFFOptimizeMolecule(mol)
                    mol = Chem.RemoveHs(mol)
                except:
                    AllChem.Compute2DCoords(mol)

                afm, adj, dist = featurize_mol(mol, self._add_dummy_node, self._one_hot_formal_charge)
                mask = np.sum(np.abs(dist), axis=-1) != 0
                features_list.append(afm)
                adjacency_list.append(adj)
                distance_list.append(dist)
                mask_list.append(mask)
            except ValueError as e:
                logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

        if padding:
            max_size = max(m.shape[0] for m in adjacency_list)
            adjacency_list = [pad_array(adj, shape=(max_size, max_size)) for adj in adjacency_list]
            distance_list = [pad_array(dist, shape=(max_size, max_size)) for dist in distance_list]
            features_list = [pad_array(afm, shape=(max_size, afm.shape[1])) for afm in features_list]
            mask_list = [pad_array(mask, shape=max_size, dtype=np.bool) for mask in mask_list]

        if return_tensors == 'pt':
            adjacency_list = FloatTensor(adjacency_list)
            distance_list = FloatTensor(distance_list)
            features_list = FloatTensor(features_list)
            mask_list = BoolTensor(mask_list)

        return {'node_features': features_list,
                'adjacency_matrix': adjacency_list,
                'distance_matrix': distance_list,
                'mask': mask_list}


def featurize_mol(mol, add_dummy_node, one_hot_formal_charge):
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


def get_atom_features(atom, one_hot_formal_charge=True):
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


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


def pad_array(array, *, shape, dtype=np.float32):
    """Pad an array with zeros.

    Args:
        array (ndarray): An array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        An array of the given shape padded with zeros.
    """
    result = np.zeros(shape, dtype=dtype)
    slices = tuple(slice(s) for s in array.shape)
    result[slices] = array
    return result
