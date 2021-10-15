from dataclasses import dataclass
from typing import List

from .configuration_api import PretrainedConfigMixin

MAT_CONFIG_ARCH = {
    'mat_masking_200k': 'https://drive.google.com/uc?id=1OundupS0xJz3c-qvPqTHuAAxLRYHclQF',
    'mat_masking_2M': 'https://drive.google.com/uc?id=1OundupS0xJz3c-qvPqTHuAAxLRYHclQF',
    'mat_masking_20M': 'https://drive.google.com/uc?id=1OundupS0xJz3c-qvPqTHuAAxLRYHclQF'
}


@dataclass
class MatConfig(PretrainedConfigMixin):
    d_atom: int = 36
    d_model: int = 1024
    init_type: str = 'uniform'
    dropout: float = 0.0

    encoder_n_layers: int = 8
    encoder_n_attn_heads: int = 16
    distance_matrix_kernel: str = 'exp'
    lambda_attention: float = 0.33
    lambda_distance: float = 0.33

    ffn_activation: str = 'LeakyReLU'
    ffn_n_layers: int = 1
    ffn_d_hidden: int = 0

    generator_aggregation: str = 'mean'
    generator_n_layers: int = 1
    generator_n_outputs: int = 1
    generator_features_generators: List[str] = None
    generator_d_generated_features: int = 0

    @classmethod
    def _get_archive_dict(cls) -> dict:
        return MAT_CONFIG_ARCH
