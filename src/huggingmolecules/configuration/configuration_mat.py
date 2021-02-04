from dataclasses import dataclass

from .configuration_api import PretrainedConfigMixin

MAT_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING = {
    'mat_masking_2M_old': './pretrained/mat/configs/mat_model_config_old.json',
    'mat_masking_200k': './pretrained/mat/configs/mat_model_config.json',
    'mat_masking_2M': './pretrained/mat/configs/mat_model_config.json',
    'mat_masking_20M': './pretrained/mat/configs/mat_model_config.json'
}


@dataclass
class MatConfig(PretrainedConfigMixin):
    d_atom: int = 36
    d_model: int = 1024
    N: int = 8
    h: int = 16
    N_dense: int = 1
    lambda_attention: float = 0.33
    lambda_distance: float = 0.33
    lin_factor: float = 1
    distance_matrix_kernel: str = 'exp'
    dropout: float = 0.0
    aggregation_type: str = 'mean'
    n_generator_layers: int = 1
    n_output: int = 1
    trainable_lambda: bool = False
    integrated_distances: bool = False
    use_edge_features: bool = False
    control_edges: bool = False
    scale_norm: bool = False
    use_adapter: bool = False
    init_type: str = 'uniform'
    add_dummy_node: bool = True
    one_hot_formal_charge: bool = True
    one_hot_formal_charge_range: list = (-5, -4, -4, -2, -1, 0, 1, 2, 3, 4, 5)

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str):
        return MAT_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING.get(pretrained_name, None)
