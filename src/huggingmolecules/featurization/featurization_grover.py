from dataclasses import dataclass
from typing import *

import torch
from rdkit import Chem

from .featurization_api import RecursiveToDeviceMixin, PretrainedFeaturizerMixin
from .featurization_common_utils import stack_y, generate_additional_features, stack_generated_features
from .featurization_grover_utils import build_atom_features, build_bond_features_and_mappings
from ..configuration import GroverConfig


@dataclass
class GroverMoleculeEncoding:
    f_atoms: list
    f_bonds: list
    a2b: list
    b2a: list
    b2revb: List
    n_atoms: int
    n_bonds: int
    generated_features: Optional[List[float]]
    y: Optional[float]


@dataclass
class GroverBatchEncoding(RecursiveToDeviceMixin):
    f_atoms: torch.FloatTensor
    f_bonds: torch.FloatTensor
    a2b: torch.LongTensor
    b2a: torch.LongTensor
    b2revb: torch.LongTensor
    a2a: torch.LongTensor
    a_scope: torch.LongTensor
    b_scope: torch.LongTensor
    generated_features: Optional[torch.FloatTensor]
    y: Optional[torch.FloatTensor]
    batch_size: int

    def __len__(self):
        return self.batch_size

    def get_components(self):
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.a2a


class GroverFeaturizer(PretrainedFeaturizerMixin[GroverMoleculeEncoding, GroverBatchEncoding, GroverConfig]):
    @classmethod
    def _get_config_cls(cls) -> Type[GroverConfig]:
        return GroverConfig

    def __init__(self, config: GroverConfig):
        super().__init__(config)
        self.atom_fdim = config.d_atom
        self.bond_fdim = config.d_bond + config.d_atom

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> GroverMoleculeEncoding:
        mol = Chem.MolFromSmiles(smiles)

        atom_features = build_atom_features(mol)
        bond_features, a2b, b2a, b2revb = build_bond_features_and_mappings(mol, atom_features)
        generated_features = generate_additional_features(mol, self.config.ffn_features_generators)

        return GroverMoleculeEncoding(f_atoms=atom_features,
                                      f_bonds=bond_features,
                                      a2b=a2b,
                                      b2a=b2a,
                                      b2revb=b2revb,
                                      n_atoms=len(atom_features),
                                      n_bonds=len(bond_features),
                                      generated_features=generated_features,
                                      y=y)

    def _collate_encodings(self, encodings: List[GroverMoleculeEncoding]) -> GroverBatchEncoding:
        # Start n_atoms and n_bonds at 1 b/c zero padding
        n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond

        for mol_graph in encodings:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(n_atoms + mol_graph.b2a[b])
                b2revb.append(n_bonds + mol_graph.b2revb[b])

            a_scope.append((n_atoms, mol_graph.n_atoms))
            b_scope.append((n_bonds, mol_graph.n_bonds))
            n_atoms += mol_graph.n_atoms
            n_bonds += mol_graph.n_bonds

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        f_atoms = torch.FloatTensor(f_atoms)
        f_bonds = torch.FloatTensor(f_bonds)
        a2b = torch.LongTensor([a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)])
        b2a = torch.LongTensor(b2a)
        b2revb = torch.LongTensor(b2revb)
        a2a = b2a[a2b]  # only needed if using atom messages
        a_scope = torch.LongTensor(a_scope)
        b_scope = torch.LongTensor(b_scope)

        return GroverBatchEncoding(f_atoms=f_atoms,
                                   f_bonds=f_bonds,
                                   a2a=a2a,
                                   a2b=a2b,
                                   b2a=b2a,
                                   b2revb=b2revb,
                                   a_scope=a_scope,
                                   b_scope=b_scope,
                                   y=stack_y(encodings),
                                   generated_features=stack_generated_features(encodings),
                                   batch_size=len(encodings))
