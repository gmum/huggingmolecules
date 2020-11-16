from dataclasses import dataclass

from .configuration_api import PretrainedConfigMixin

MAT_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING = {
    'mat-base-freesolv': './saved/mat-base-freesolv-config',
    'mat-base-freesolv-tests': '../saved/mat-base-freesolv-config'
}


@dataclass
class MatConfig(PretrainedConfigMixin):
    d_atom: int = 28
    d_model: int = 1024
    N: int = 8
    h: int = 16
    N_dense: int = 1
    lambda_attention: float = 0.33
    lambda_distance: float = 0.33
    leaky_relu_slope: float = 0.1
    dense_output_nonlinearity: str = 'relu'
    distance_matrix_kernel: str = 'exp'
    dropout: float = 0.0,
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

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str):
        return MAT_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING.get(pretrained_name, None)
