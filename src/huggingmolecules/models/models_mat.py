"""
This implementation is adapted from
https://github.com/ardigen/MAT/blob/master/src/transformer.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.huggingmolecules.configuration.configuration_mat import MatConfig
from src.huggingmolecules.featurization.featurization_mat import MatBatchEncoding, MatFeaturizer
from .models_api import PretrainedModelBase
from .models_common_utils import MultiHeadedAttention, PositionwiseFeedForward, Embedding, Encoder, Generator

MAT_PRETRAINED_NAME_TO_WEIGHTS_ARCH_MAPPING = {
    'mat_masking_2M_old': './pretrained/mat/weights/mat_masking_2M_old.pt',
    'mat_masking_200k': './pretrained/mat/weights/mat_masking_200k.pt',
    'mat_masking_2M': './pretrained/mat/weights/mat_masking_2M.pt',
    'mat_masking_20M': './pretrained/mat/weights/mat_masking_20M.pt'
}


class MatModel(PretrainedModelBase[MatBatchEncoding, MatConfig]):
    @classmethod
    def get_config_cls(cls):
        return MatConfig

    @classmethod
    def get_featurizer_cls(cls):
        return MatFeaturizer

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str):
        return MAT_PRETRAINED_NAME_TO_WEIGHTS_ARCH_MAPPING.get(pretrained_name, None)

    def __init__(self, config: MatConfig):
        super().__init__(config)

        # Embedding
        self.src_embed = Embedding(d_input=config.d_atom,
                                   d_output=config.d_model,
                                   dropout=config.dropout)

        # Encoder
        attention = MatAttention(config)
        sa_layer = MultiHeadedAttention(h=config.encoder_n_attn_heads,
                                        d_model=config.d_model,
                                        dropout=config.dropout,
                                        attention=attention)
        ff_layer = PositionwiseFeedForward(d_input=config.d_model,
                                           activation=config.ffn_activation,
                                           n_layers=config.ffn_n_layers,
                                           dropout=config.dropout)

        self.encoder = Encoder(sa_layer=sa_layer,
                               ff_layer=ff_layer,
                               d_model=config.d_model,
                               dropout=config.dropout,
                               n_layers=config.encoder_n_layers)

        # Generator
        self.generator = Generator(d_model=config.d_model,
                                   aggregation_type=config.generator_aggregation,
                                   d_output=config.generator_n_outputs,
                                   n_layers=config.generator_n_layers,
                                   dropout=config.dropout)

        # Initialization
        self.init_weights(config.init_type)

    def forward(self, batch: MatBatchEncoding):
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = self.src_embed(batch.node_features)
        encoded = self.encoder(embedded, batch_mask,
                               adj_matrix=batch.adjacency_matrix,
                               distance_matrix=batch.distance_matrix)
        output = self.generator(encoded, batch_mask)
        return output


# Attention

class MatAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.distance_matrix_kernel == "softmax":
            self.distance_matrix_kernel = lambda x: torch.softmax(-x, dim=-1)
        elif config.distance_matrix_kernel == "exp":
            self.distance_matrix_kernel = lambda x: torch.exp(-x)

        self.lambda_attention = config.lambda_attention
        self.lambda_distance = config.lambda_distance
        self.lambda_adjacency = 1. - self.lambda_attention - self.lambda_distance

    def forward(self, query, key, value, mask, dropout,
                adj_matrix, distance_matrix,
                eps=1e-6, inf=1e12):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0, -inf)
        p_attn = F.softmax(scores, dim=-1)

        # Prepare adjacency matrix
        adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
        p_adj = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        # Prepare distances matrix
        distance_matrix = distance_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
        distance_matrix = self.distance_matrix_kernel(distance_matrix)
        p_dist = distance_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        p_weighted = self.lambda_attention * p_attn + self.lambda_distance * p_dist + self.lambda_adjacency * p_adj
        p_weighted = dropout(p_weighted)

        return torch.matmul(p_weighted, value), p_attn
