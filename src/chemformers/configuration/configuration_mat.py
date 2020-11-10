from .configuration_api import PretrainedConfigMixin

MAT_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING = {
    'mat-base-freesolv': '/home/panjan/Desktop/GMUM/chemformers/saved/mat-base-freesolv-config'
}


class MatConfig(PretrainedConfigMixin["MatConfig"]):
    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name):
        return MAT_PRETRAINED_NAME_TO_CONFIG_ARCH_MAPPING.get(pretrained_name, None)

    def __init__(self, d_atom=28, d_model=1024, N=8, h=16, N_dense=1, lambda_attention=0.33, lambda_distance=0.33,
                 leaky_relu_slope=0.1, dense_output_nonlinearity='relu', distance_matrix_kernel='exp', dropout=0.0,
                 aggregation_type='mean', n_generator_layers=1, n_output=1, trainable_lambda=False,
                 integrated_distances=False, use_edge_features=False, control_edges=False, scale_norm=False,
                 use_adapter=False, init_type='uniform'):
        super().__init__()
        self.d_atom = d_atom
        self.d_model = d_model
        self.N = N
        self.h = h
        self.N_dense = N_dense
        self.lambda_attention = lambda_attention
        self.lambda_distance = lambda_distance
        self.leaky_relu_slope = leaky_relu_slope
        self.dense_output_nonlinearity = dense_output_nonlinearity
        self.distance_matrix_kernel = distance_matrix_kernel
        self.dropout = dropout
        self.aggregation_type = aggregation_type
        self.n_generator_layers = n_generator_layers
        self.n_output = n_output
        self.trainable_lambda = trainable_lambda
        self.integrated_distances = integrated_distances
        self.use_edge_features = use_edge_features
        self.control_edges = control_edges
        self.control_edges = control_edges
        self.scale_norm = scale_norm
        self.use_adapter = use_adapter
        self.init_type = init_type
