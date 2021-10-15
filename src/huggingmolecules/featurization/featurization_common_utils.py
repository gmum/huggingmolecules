from typing import List, Optional, Union

import numpy as np
import torch
from rdkit import Chem

from .featurization_api import T_BatchEncoding
from .featurization_features_generators import get_features_generator


def one_hot_vector(value: Union[float, int],
                   choices: List[Union[float, int]],
                   extra_category: bool = False) -> List[Union[float, int]]:
    encoding = [0] * len(choices)
    if extra_category:
        encoding.append(0)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def stack_y(encodings: List[T_BatchEncoding]) -> Optional[torch.FloatTensor]:
    return stack_y_list([e.y for e in encodings])


def stack_generated_features(encodings: List[T_BatchEncoding]) -> Optional[torch.FloatTensor]:
    if encodings[0].generated_features is not None:
        return torch.stack([torch.tensor(mol.generated_features) for mol in encodings]).float()
    else:
        return None


def stack_y_list(y_list: List[float]) -> Optional[torch.FloatTensor]:
    if any(y is None for y in y_list):
        return None
    else:
        return torch.stack([torch.tensor(y).float() for y in y_list]).unsqueeze(1)


def generate_additional_features(mol: Chem.Mol, features_generators: List[str]) -> List[float]:
    if features_generators is None:
        return None
    generated_features = []
    dummy = Chem.MolFromSmiles('C')
    for generator_name in features_generators:
        generator = get_features_generator(generator_name)
        if mol.GetNumHeavyAtoms() > 0:
            generated_features.extend(generator(mol))
        else:
            generated_features.extend(np.zeros(len(generator(dummy))))
    return generated_features
