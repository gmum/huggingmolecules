from dataclasses import dataclass
from typing import List

from .configuration_api import PretrainedConfigMixin

RMAT_CONFIG_ARCH = {
    'rmat_4M': 'https://drive.google.com/uc?id=1VBou9SzOnLPAC6NdAKB1nYVZ23ninvMY',
    'rmat_4M_rdkit': 'https://drive.google.com/uc?id=18zVAMupEdSfBGebD5SFOWFVaiFhDOaYd'
}

@dataclass
class RMatConfig(PretrainedConfigMixin):
    d_atom: int = 37
    d_edge: int = 46
    d_model: int = 768
    init_type: str = 'uniform'
    dropout: float = 0.0

    envelope_num_radial: int = 32
    envelope_cutoff: float = 20.0
    envelope_exponent: float = 5.0

    encoder_n_layers: int = 10
    encoder_n_attn_heads: int = 12

    ffn_activation: str = 'LeakyReLU'
    ffn_n_layers: int = 2
    ffn_d_hidden: int = 1536
    ffn_d_output: int = 768

    generator_aggregation: str = 'grover'
    generator_n_layers: int = 1
    generator_d_outputs: int = 1
    generator_features_generators: List[str] = None
    generator_d_generated_features: int = 0

    @classmethod
    def _get_archive_dict(cls) -> dict:
        return RMAT_CONFIG_ARCH
