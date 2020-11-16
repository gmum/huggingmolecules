from dataclasses import dataclass

from .configuration_api import PretrainedConfigMixin

GROVER_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING = {
    'grover-base-whatever': './saved/grover-base-whatever-config'
}


@dataclass
class GroverConfig(PretrainedConfigMixin):
    model_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    num_layers_dympnn: int = 2
    num_hops_dympnn: int = 2
    n_f_atom: int = 133
    n_f_bond: int = 14
    readout_hidden_dim: int = 128
    readout_num_heads: int = 4
    head_hidden_dim: int = 13
    head_num_layers: int = 3
    output_dim: int = 1

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str):
        return GROVER_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING.get(pretrained_name, None)
