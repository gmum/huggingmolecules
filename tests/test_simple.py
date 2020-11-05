import unittest
from src import *
from transformers import BertPreTrainedModel, BertConfig

class MatPreparerTest(unittest.TestCase):

    def test_single(self):
        featurizer = MatFeaturizer()
        input = featurizer(["CO", "CSC"], padding=False)
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
        featurizer = MatFeaturizer()
        input = featurizer(["CO", "CSC"], padding=True)
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

class MatConfigTest(unittest.TestCase):

    def test_from_pretrained(self):
        config = MatConfig.from_pretrained('mat-base-freesolv')
        print(config)


class MatModelTest(unittest.TestCase):

    def test_explicit(self):
        featurizer = MatFeaturizer()
        input = featurizer(["CC(C)Cl", "CCCBr"], padding=True, return_tensors='pt')

        config = MatConfig()
        model = MatModel(config)
        model.load_weights('../mat-base-freesolv')

        output = model(**input)
        print(output)

    def test_errors(self):
        pass

    def test_from_pretrained(self):
        featurizer = MatFeaturizer()
        input = featurizer(["CC(C)Cl", "CCCBr"], padding=True, return_tensors='pt')

        model = MatModel.from_pretrained('mat-base-freesolv')

        output = model(**input)
        print(output)