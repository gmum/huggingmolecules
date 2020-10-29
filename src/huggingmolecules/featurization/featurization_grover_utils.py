"""
This implementation is adapted from
https://github.com/tencent-ailab/grover/main/grover/data/molgraph.py
"""

from typing import List, Union, Tuple, Any

from rdkit import Chem

from .featurization_common_utils import one_hot_vector


def get_atom_features(
        atom: Chem.rdchem.Atom,
        hydrogen_acceptor_match,
        hydrogen_donor_match,
        acidic_match,
        basic_match,
        ring_info) -> List[Union[bool, int, float]]:
    features = []

    features += one_hot_vector(atom.GetAtomicNum() - 1, [i for i in range(100)], extra_category=True)
    features += one_hot_vector(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5], extra_category=True)
    features += one_hot_vector(atom.GetFormalCharge(), [-1, -2, 1, 2, 0], extra_category=True)
    features += one_hot_vector(int(atom.GetChiralTag()), [0, 1, 2, 3], extra_category=True)
    features += one_hot_vector(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4], extra_category=True)
    features += one_hot_vector(int(atom.GetHybridization()), [2, 3, 4, 5, 6], extra_category=True)
    features += [1 if atom.GetIsAromatic() else 0]
    features += [atom.GetMass() * 0.01]
    features += one_hot_vector(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6], extra_category=True)

    atom_idx = atom.GetIdx()
    features += [atom_idx in hydrogen_acceptor_match]
    features += [atom_idx in hydrogen_donor_match]
    features += [atom_idx in acidic_match]
    features += [atom_idx in basic_match]
    features += [
        ring_info.IsAtomInRingOfSize(atom_idx, 3),
        ring_info.IsAtomInRingOfSize(atom_idx, 4),
        ring_info.IsAtomInRingOfSize(atom_idx, 5),
        ring_info.IsAtomInRingOfSize(atom_idx, 6),
        ring_info.IsAtomInRingOfSize(atom_idx, 7),
        ring_info.IsAtomInRingOfSize(atom_idx, 8)
    ]

    return features


def get_bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    bond_type = bond.GetBondType()
    features = [
        0,  # bond is not None
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bond_type is not None else 0),
        (bond.IsInRing() if bond_type is not None else 0)
    ]
    features += one_hot_vector(int(bond.GetStereo()), list(range(6)), extra_category=True)
    return features


def build_atom_features(mol: Chem.Mol) -> List[Any]:
    hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
    hydrogen_acceptor = Chem.MolFromSmarts(
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
        "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
    acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
    basic = Chem.MolFromSmarts(
        "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
        "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())
    ring_info = mol.GetRingInfo()

    return [get_atom_features(atom,
                              hydrogen_acceptor_match,
                              hydrogen_donor_match,
                              acidic_match,
                              basic_match,
                              ring_info) for atom in mol.GetAtoms()]


def build_bond_features_and_mappings(mol: Chem.Mol, f_atoms: List) -> Tuple[list, list, list, list]:
    f_bonds = []
    a2b = [[] for _ in range(mol.GetNumAtoms())]  # mapping from atom index to incoming bond indices
    b2a = []  # mapping from bond index to the index of the atom the bond is coming from
    b2revb = []  # mapping from bond index to the index of the reverse bond

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        f_bond = get_bond_features(bond)

        f_bonds.append(f_atoms[a1] + f_bond)
        f_bonds.append(f_atoms[a2] + f_bond)

        # Update index mappings
        b1 = len(f_bonds) - 2
        b2 = b1 + 1
        b2a.append(a1)
        b2a.append(a2)
        a2b[a2].append(b1)  # b1 = a1 --> a2
        a2b[a1].append(b2)  # b2 = a2 --> a1
        b2revb.append(b2)
        b2revb.append(b1)

    return f_bonds, a2b, b2a, b2revb
