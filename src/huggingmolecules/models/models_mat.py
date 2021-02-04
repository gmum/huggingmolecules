"""
This implementation is adapted from
https://github.com/ardigen/MAT/blob/master/src/transformer.py
"""

import math
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.huggingmolecules.configuration.configuration_mat import MatConfig
from src.huggingmolecules.featurization.featurization_mat import MatBatchEncoding, MatFeaturizer
from .models_api import PretrainedModelBase
from .models_common_utils import clones

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

        self.src_embed = Embeddings(config.d_atom, config.d_model, config.dropout)
        ff = PositionwiseFeedForward(config.d_model, config.N_dense, config.dropout, config.lin_factor)
        attn = MultiHeadedAttention(config)
        self.encoder = Encoder(attn, ff, config.d_model, config.dropout, config.N)
        self.generator = Generator(config.d_model, config.aggregation_type, config.n_output, config.n_generator_layers,
                                   config.dropout)

        self.init_weights(config.init_type)

    def forward(self, batch: MatBatchEncoding):
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = self.src_embed(batch.node_features)
        encoded = self.encoder(embedded, batch_mask, adj_matrix=batch.adjacency_matrix,
                               distance_matrix=batch.distance_matrix, edges_att=None)
        output = self.generator(encoded, batch_mask)
        return output


# Embeddings

class Embeddings(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))


# Encoder


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, self_attn, feed_forward, size, dropout, N):
        super(Encoder, self).__init__()
        layer = EncoderLayer(self_attn, feed_forward, size, dropout)
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask, **kwargs):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask, **kwargs)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, self_attn, feed_forward, size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, **kwargs):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask, **kwargs))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


# Attention

class EdgeFeaturesLayer(nn.Module):
    def __init__(self, d_model, d_edge, h, dropout):
        super(EdgeFeaturesLayer, self).__init__()
        assert d_model % h == 0
        d_k = d_model // h
        self.linear = nn.Linear(d_edge, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.25)

    def forward(self, x):
        p_edge = x.permute(0, 2, 3, 1)
        p_edge = self.linear(p_edge).permute(0, 3, 1, 2)
        return torch.relu(p_edge)


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        assert config.d_model % config.h == 0
        super(MultiHeadedAttention, self).__init__()

        self.d_k = config.d_model // config.h
        self.h = config.h

        self.linears = clones(nn.Linear(config.d_model, config.d_model), 4)
        self.dropout = nn.Dropout(config.dropout)
        if config.distance_matrix_kernel == "softmax":
            self.distance_matrix_kernel = lambda x: torch.softmax(-x, dim=-1)
        elif config.distance_matrix_kernel == "exp":
            self.distance_matrix_kernel = lambda x: torch.exp(-x)

        self.lambda_attention = config.lambda_attention
        self.lambda_distance = config.lambda_distance
        self.lambda_adjacency = 1. - self.lambda_attention - self.lambda_distance
        self.attn = None

    def forward(self, query, key, value, adj_matrix, distance_matrix, edges_att, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batches = query.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn, _ = self.attention(query, key, value,
                                         adj_matrix, distance_matrix, mask)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self,
                  query, key, value,
                  adj_matrix, distance_matrix, mask,
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
        p_weighted = self.dropout(p_weighted)

        atoms_featrues = torch.matmul(p_weighted, value)
        return atoms_featrues, p_weighted, p_attn


# Conv 1x1 aka Positionwise feed forward

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, N_dense, dropout, lin_factor):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        if N_dense == 1:
            self.linears = [nn.Linear(d_model, d_model)]
        else:
            self.linears = [nn.Linear(d_model, d_model * lin_factor)] + \
                           [nn.Linear(d_model * lin_factor, d_model * lin_factor) for _ in range(N_dense - 2)] + \
                           [nn.Linear(d_model * lin_factor, d_model)]

        self.linears = nn.ModuleList(self.linears)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.nonlinearity = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        if self.N_dense == 0:
            return x
        elif self.N_dense == 1:
            return self.dropout[0](self.nonlinearity(self.linears[0](x)))
        else:
            for i in range(self.N_dense - 1):
                x = self.dropout[i](self.nonlinearity(self.linears[i](x)))
            return self.linears[-1](x)


# Generator

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, aggregation_type='grover', n_output=1, n_layers=1, dropout=0.0,
                 attn_hidden=128, attn_out=4):
        super(Generator, self).__init__()
        if aggregation_type == 'grover':
            self.att_net = nn.Sequential(
                nn.Linear(d_model, attn_hidden, bias=False),
                nn.Tanh(),
                nn.Linear(attn_hidden, attn_out, bias=False),
            )
            d_model *= attn_out

        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            self.proj = []
            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(d_model, attn_hidden))
                self.proj.append(nn.LeakyReLU(negative_slope=0.1))
                self.proj.append(nn.LayerNorm(attn_hidden))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(attn_hidden, n_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask
        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=(1))
            out_avg_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'grover':
            out_attn = self.att_net(out_masked)
            out_attn = out_attn.masked_fill(mask == 0, -1e9)
            out_attn = F.softmax(out_attn, dim=1)
            out_avg_pooling = torch.matmul(torch.transpose(out_attn, -1, -2), out_masked)
            out_avg_pooling = out_avg_pooling.view(out_avg_pooling.size(0), -1)
        elif self.aggregation_type == 'contextual':
            out_avg_pooling = x
        projected = self.proj(out_avg_pooling)
        return projected
