from dataclasses import dataclass

from .configuration_api import PretrainedConfigMixin

MAT_CONFIG_ARCH = {
    'mat_masking_200k': '/home/panjan/Desktop/GMUM/huggingmolecules/pretrained/mat/configs/mat_model_config.json',
    'mat_masking_2M': '/home/panjan/Desktop/GMUM/huggingmolecules/pretrained/mat/configs/mat_model_config.json',
    'mat_masking_20M': '/home/panjan/Desktop/GMUM/huggingmolecules/pretrained/mat/configs/mat_model_config.json'
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

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        return MAT_CONFIG_ARCH.get(pretrained_name, None)
