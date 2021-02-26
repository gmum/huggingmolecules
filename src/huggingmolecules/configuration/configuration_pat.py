from dataclasses import dataclass

from .configuration_api import PretrainedConfigMixin

PAT_CONFIG_ARCH = {
    'pat_test': '/home/panjan/Desktop/GMUM/huggingmolecules/pretrained/pat/configs/pat_test.json',
}


@dataclass
class PatConfig(PretrainedConfigMixin):
    d_atom: int = 37
    d_edge: int = 40
    d_model: int = 1024
    init_type: str = 'normal'
    dropout: float = 0.0

    envelope_num_radial: int = 32
    envelope_cutoff: float = 20.0
    envelope_exponent: float = 5.0

    encoder_n_layers: int = 8
    encoder_n_attn_heads: int = 16

    ffn_activation: str = 'LeakyReLU'
    ffn_n_layers: int = 1
    ffn_d_hidden: int = 0

    generator_aggregation: str = 'mean'
    generator_n_layers: int = 1
    generator_d_outputs: int = 1

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        return PAT_CONFIG_ARCH.get(pretrained_name, None)
