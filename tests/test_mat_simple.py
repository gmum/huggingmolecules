import unittest
import numpy as np
from src.chemformers import MatFeaturizer, MatConfig, MatModel


class MatFeaturizerTest(unittest.TestCase):

    def test_single(self):
        featurizer = MatFeaturizer()
        batch = featurizer.__call__(["CO"])
        # batch = featurizer(["CO"])  # it doesn't type well for some reason

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

        assert np.allclose(exp_node_features, batch.node_features)
        assert np.allclose(exp_adj_matrix, batch.adjacency_matrix)
        assert np.allclose(exp_dist_matrix, batch.distance_matrix)
        assert np.allclose(exp_mask, batch.batch_mask)

    def test_padded(self):
        featurizer = MatFeaturizer()
        batch = featurizer.__call__(["CO", "CSC"])

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

        assert np.allclose(exp_node_features, batch.node_features[0])
        assert np.allclose(exp_adj_matrix, batch.adjacency_matrix[0])
        assert np.allclose(exp_dist_matrix, batch.distance_matrix[0])
        assert np.allclose(exp_mask, batch.batch_mask[0])


class MatConfigTest(unittest.TestCase):

    def test_from_pretrained(self):
        config = MatConfig.from_pretrained('mat-base-freesolv')
        print(config)


class MatModelTest(unittest.TestCase):

    def test_errors(self):
        pass

    def test_from_pretrained(self):
        featurizer = MatFeaturizer()
        batch = featurizer.__call__(["CC(C)Cl", "CCCBr"])

        model = MatModel.from_pretrained('mat-base-freesolv')
        output = model(batch)
        print(output)
