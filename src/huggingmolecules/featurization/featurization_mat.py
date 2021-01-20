import logging
from dataclasses import dataclass
from typing import *

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.huggingmolecules import MatConfig
from .featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from .featurization_mat_utils import featurize_mol, pad_array


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
    batch_mask: torch.BoolTensor
    y: Optional[torch.FloatTensor]
    batch_size: int

    def __len__(self):
        return self.batch_size


class MatFeaturizer(PretrainedFeaturizerMixin[MatMoleculeEncoding, MatBatchEncoding]):
    @classmethod
    def get_config_cls(cls):
        return MatConfig

    def __init__(self, config: MatConfig):
        self._add_dummy_node = config.add_dummy_node
        self._one_hot_formal_charge = config.one_hot_formal_charge
        self._one_hot_formal_charge_range = config.one_hot_formal_charge_range

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> MatMoleculeEncoding:
        try:
            mol = Chem.MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=10)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(mol, self._add_dummy_node, self._one_hot_formal_charge,
                                           self._one_hot_formal_charge_range)
            afm = torch.tensor(afm).float()
            adj = torch.tensor(adj).float()
            dist = torch.tensor(dist).float()
            y = None if y is None else torch.tensor(y).float()
            return MatMoleculeEncoding(node_features=afm, adjacency_matrix=adj, distance_matrix=dist, y=y)
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    def _collate_encodings(self, encodings: List[MatMoleculeEncoding]) -> MatBatchEncoding:
        adjacency_list, distance_list, features_list, batch_mask_list, y_list = [], [], [], [], []
        max_size = max(e.adjacency_matrix.shape[0] for e in encodings)
        for e in encodings:
            adjacency_list.append(pad_array(e.adjacency_matrix, size=(max_size, max_size)))
            distance_list.append(pad_array(e.distance_matrix, size=(max_size, max_size)))
            features_list.append(pad_array(e.node_features, size=(max_size, e.node_features.shape[1])))
            batch_mask = torch.sum(torch.abs(e.node_features), dim=-1) != 0
            batch_mask_list.append(pad_array(batch_mask, size=(max_size,), dtype=torch.bool))
            y_list.append(e.y)

        return MatBatchEncoding(node_features=torch.stack(features_list).float(),
                                adjacency_matrix=torch.stack(adjacency_list).float(),
                                distance_matrix=torch.stack(distance_list).float(),
                                batch_mask=torch.stack(batch_mask_list).bool(),
                                y=None if None in y_list else torch.stack(y_list).float().view(-1, 1),
                                batch_size=len(features_list))
