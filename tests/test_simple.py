import unittest
from src import *


class MatPreparerTest(unittest.TestCase):

    def test_single(self):
        preparer = MatPreparer()
        input = preparer.encode(["CO", "CSC"], padding=False)
        node_features = input['node_features'][0]
        adj_matrix = input['adjacency_matrix'][0]
        dist_matrix = input['distance_matrix'][0]
        mask = input['mask'][0]

        exp_node_features = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        exp_adj_matrix = np.array([[0.0, 0.0, 0.0],
                                   [0.0, 1.0, 1.0],
                                   [0.0, 1.0, 1.0]])
        exp_dist_matrix = np.array([[1000000.0, 1000000.0, 1000000.0],
                                    [1000000.0, 0.0, 1.3984568119049072],
                                    [1000000.0, 1.3984568119049072, 0.0]])
        exp_mask = np.array([True, True, True])

        assert np.allclose(exp_node_features, node_features)
        assert np.allclose(exp_adj_matrix, adj_matrix)
        assert np.allclose(exp_dist_matrix, dist_matrix)
        assert np.allclose(exp_mask, mask)

    def test_padded(self):
        preparer = MatPreparer()
        input = preparer.encode(["CO", "CSC"], padding=True)
        node_features = input['node_features'][0]
        adj_matrix = input['adjacency_matrix'][0]
        dist_matrix = input['distance_matrix'][0]
        mask = input['mask'][0]

        exp_node_features = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        exp_adj_matrix = np.array([[0.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 1.0, 0.0],
                                   [0.0, 1.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0]])
        exp_dist_matrix = np.array([[1000000.0, 1000000.0, 1000000.0, 0.0],
                                    [1000000.0, 0.0, 1.3984568119049072, 0.0],
                                    [1000000.0, 1.3984568119049072, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]])
        exp_mask = np.array([True, True, True, False])

        assert np.allclose(exp_node_features, node_features)
        assert np.allclose(exp_adj_matrix, adj_matrix)
        assert np.allclose(exp_dist_matrix, dist_matrix)
        assert np.allclose(exp_mask, mask)


class MatForRegressionTest(unittest.TestCase):

    def test_single(self):
        model_params = {
            'd_atom': 28,
            'd_model': 1024,
            'N': 8,
            'h': 16,
            'N_dense': 1,
            'lambda_attention': 0.33,
            'lambda_distance': 0.33,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'relu',
            'distance_matrix_kernel': 'exp',
            'dropout': 0.0,
            'aggregation_type': 'mean'
        }

        model = make_model(**model_params)

        pretrained_name = '../pretrained_weights.pt'  # This file should be downloaded first (See README.md).
        pretrained_state_dict = torch.load(pretrained_name)
        model_state_dict = model.state_dict()
        for name, param in pretrained_state_dict.items():
            if 'generator' in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

        preparer = MatPreparer()
        input = preparer.encode(["CC(C)Cl", "CCCBr"], padding=True, return_tensors='pt')
        output = model(**input)
        print(output)

