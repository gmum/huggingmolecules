import unittest
from src import *
from transformers import BertPreTrainedModel, BertConfig
import torch_geometric
from torch_geometric.data import Batch, DataLoader


class GroverFeaturizerTest(unittest.TestCase):

    def test_single(self):
        featurizer = GroverFeaturizer()
        input = featurizer(["CO", "CSC"], padding=False)
        print(input)


class GroverModelTest(unittest.TestCase):

    def test_explicit(self):
        featurizer = GroverFeaturizer()
        input = featurizer(["CO", "CSC"], padding=False)
        batch = prepare_batch(input)

        config = {'model_dim': 256,
                  'num_layers': 2,
                  'num_heads': 4,
                  'dropout': 0.1,
                  'num_layers_dympnn': 2,
                  'num_hops_dympnn': 2,
                  'n_f_atom': 133,
                  'n_f_bond': 14,
                  'readout_hidden_dim': 128,
                  'readout_num_heads': 4,
                  'head_hidden_dim': 13,
                  'head_num_layers': 3,
                  'output_dim': 1
                  }

        model = Grover(**config)
        output = model(batch)
        print(output)

def prepare_batch(data_list: List[Data]) -> Batch:
    data_loader = DataLoader(data_list, len(data_list))
    return [batch for batch in data_loader][0]