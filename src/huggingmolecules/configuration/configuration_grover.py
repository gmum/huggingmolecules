from dataclasses import dataclass

from .configuration_api import PretrainedConfigMixin

GROVER_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING = {
    'grover_base': './pretrained/grover/grover_base_config.json',
    'grover_large': './pretrained/grover/grover_large_config.json'
}


@dataclass
class GroverConfig(PretrainedConfigMixin):
    hidden_size: int = 128
    self_attention: bool = False
    attn_hidden: int = 4
    attn_out: int = 128
    bond_fdim: int = 14
    atom_fdim: int = 151
    embedding_output_type: str = 'both'
    activation: str = 'PReLU'
    num_mt_block: int = 1
    num_attn_head: int = 4
    bias: bool = False
    features_only: bool = False
    features_dim: int = 0
    ffn_num_layers: int = 2
    output_size: int = 1  # shouldn't be changed
    ffn_hidden_size: int = 128  # defaults to hidden_size
    dropout: float = 0.0
    depth: int = 6
    undirected: bool = False
    dense: bool = False
    backbone: str = 'dualtrans'

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str):
        return GROVER_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING.get(pretrained_name, None)
