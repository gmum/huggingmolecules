from dataclasses import dataclass, field
from typing import List

from .configuration_api import PretrainedConfigMixin

GROVER_CONFIG_ARCH = {
    'grover_base': 'https://drive.google.com/uc?id=1IJUedzhWSiPDdiVPKyAhv0JAsH_xqKHz',
    'grover_large': 'https://drive.google.com/uc?id=1T2ot3tF6aRR9jsveKB5ZEdvCOpt-D8Aj'
}


@dataclass
class GroverConfig(PretrainedConfigMixin):
    d_bond: int = 14
    d_atom: int = 151
    d_model: int = 800
    init_type: str = 'normal'
    backbone: str = 'dualtrans'
    activation: str = 'PReLU'
    dropout: float = 0.0

    encoder_output_type: str = 'both'
    encoder_n_blocks: int = 1
    encoder_n_attn_heads: int = 4
    encoder_attn_output_bias: bool = False

    mpn_depth: int = 6
    mpn_undirected: bool = False
    mpn_dense: bool = False

    readout_self_attention: bool = False
    readout_attn_hidden: int = None
    readout_attn_out: int = None
    readout_n_outputs: int = 1

    ffn_features_only: bool = False
    ffn_features_generators: List[str] = None
    ffn_d_generated_features: int = 0
    ffn_d_hidden: int = 128
    ffn_n_layers: int = 2

    @classmethod
    def _get_archive_dict(cls) -> dict:
        return GROVER_CONFIG_ARCH
