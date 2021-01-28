"""
This implementation is adapted from
https://github.com/ardigen/MAT/blob/master/src/transformer.py
"""

import copy
import math
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.huggingmolecules.configuration.configuration_mat import MatConfig
from src.huggingmolecules.featurization.featurization_mat import MatBatchEncoding, MatFeaturizer
from .models_api import PretrainedModelBase
from .models_mat_utils import xavier_normal_small_init_, xavier_uniform_small_init_

MAT_PRETRAINED_NAME_TO_WEIGHTS_ARCH_MAPPING = {
    'mat_masking_2M_old': './pretrained/mat/weights/mat_masking_2M_old.pt',
    'mat_masking_200k': './pretrained/mat/weights/mat_masking_200k.pt',
    'mat_masking_2M': './pretrained/mat/weights/mat_masking_2M.pt',
    'mat_masking_20M': './pretrained/mat/weights/mat_masking_20M.pt'
}


class MatModel(PretrainedModelBase[MatBatchEncoding, MatConfig]):
    def __init__(self, config: MatConfig):
        super().__init__(config)

        self.src_embed = Embeddings(config)
        self.encoder = Encoder(config)
        self.generator = Generator(config)

        self.init_weights(config.init_type)

    @classmethod
    def get_config_cls(cls):
        return MatConfig

    @classmethod
    def get_featurizer_cls(cls):
        return MatFeaturizer

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str):
        return MAT_PRETRAINED_NAME_TO_WEIGHTS_ARCH_MAPPING.get(pretrained_name, None)

    def forward(self, batch: MatBatchEncoding):
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = self.src_embed(batch.node_features)
        encoded = self.encoder(embedded, batch_mask, batch.adjacency_matrix, batch.distance_matrix, None)
        output = self.generator(encoded, batch_mask)
        return output

    def init_weights(self, init_type: str):
        for p in self.parameters():
            if p.dim() > 1:
                if init_type == 'uniform':
                    nn.init.xavier_uniform_(p)
                elif init_type == 'normal':
                    nn.init.xavier_normal_(p)
                elif init_type == 'small_normal_init':
                    xavier_normal_small_init_(p)
                elif init_type == 'small_uniform_init':
                    xavier_uniform_small_init_(p)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, config):
        super(Generator, self).__init__()
        if config.n_generator_layers == 1:
            self.proj = nn.Linear(config.d_model, config.n_output)
        else:
            self.proj = []
            for i in range(config.n_generator_layers - 1):
                self.proj.append(nn.Linear(config.d_model, config.d_model))
                self.proj.append(nn.LeakyReLU(config.leaky_relu_slope))
                self.proj.append(ScaleNorm(config.d_model) if config.scale_norm else LayerNorm(config.d_model))
                self.proj.append(nn.Dropout(config.dropout))
            self.proj.append(nn.Linear(config.d_model, config.n_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = config.aggregation_type

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask
        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=(1))
            out_avg_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_avg_pooling = out_sum
        elif self.aggregation_type == 'dummy_node':
            out_avg_pooling = out_masked[:, 0]
        projected = self.proj(out_avg_pooling)
        return projected


class PositionGenerator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 3)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = self.norm(x) * mask
        projected = self.proj(out_masked)
        return projected


### Encoder

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, config):
        super(Encoder, self).__init__()
        layer = EncoderLayer(config)
        self.layers = clones(layer, config.N)
        self.norm = ScaleNorm(layer.size) if config.scale_norm else LayerNorm(layer.size)

    def forward(self, x, mask, adj_matrix, distance_matrix, edges_att):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, adj_matrix, distance_matrix, edges_att)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, scale_norm, use_adapter):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.use_adapter = use_adapter
        self.adapter = Adapter(size, 8) if use_adapter else None

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.use_adapter:
            return x + self.dropout(self.adapter(sublayer(self.norm(x))))
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(
            SublayerConnection(config.d_model, config.dropout, config.scale_norm, config.use_adapter), 2)
        self.size = config.d_model

    def forward(self, x, mask, adj_matrix, distance_matrix, edges_att):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, distance_matrix, edges_att, mask))
        return self.sublayer[1](x, self.feed_forward)


### Attention

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


def attention(query, key, value, adj_matrix, distance_matrix, edges_att,
              mask=None, dropout=None,
              lambdas=(0.3, 0.3, 0.4), trainable_lambda=False,
              distance_matrix_kernel=None, use_edge_features=False, control_edges=False,
              eps=1e-6, inf=1e12):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0, -inf)
    p_attn = F.softmax(scores, dim=-1)

    if use_edge_features:
        adj_matrix = edges_att.view(adj_matrix.shape)

    # Prepare adjacency matrix
    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_adj = adj_matrix

    p_dist = distance_matrix

    if trainable_lambda:
        softmax_attention, softmax_distance, softmax_adjacency = lambdas.cuda()
        p_weighted = softmax_attention * p_attn + softmax_distance * p_dist + softmax_adjacency * p_adj
    else:
        lambda_attention, lambda_distance, lambda_adjacency = lambdas
        p_weighted = lambda_attention * p_attn + lambda_distance * p_dist + lambda_adjacency * p_adj

    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_featrues = torch.matmul(p_weighted, value)
    return atoms_featrues, p_weighted, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert config.d_model % config.h == 0
        # We assume d_v always equals d_k
        self.d_k = config.d_model // config.h
        self.h = config.h
        self.trainable_lambda = config.trainable_lambda
        if config.trainable_lambda:
            lambda_adjacency = 1. - config.lambda_attention - config.lambda_distance
            lambdas_tensor = torch.tensor([config.lambda_attention, config.lambda_distance, lambda_adjacency],
                                          requires_grad=True)
            self.lambdas = torch.nn.Parameter(lambdas_tensor)
        else:
            lambda_adjacency = 1. - config.lambda_attention - config.lambda_distance
            self.lambdas = (config.lambda_attention, config.lambda_distance, lambda_adjacency)

        self.linears = clones(nn.Linear(config.d_model, config.d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=config.dropout)
        if config.distance_matrix_kernel == 'softmax':
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        elif config.distance_matrix_kernel == 'exp':
            self.distance_matrix_kernel = lambda x: torch.exp(-x)
        self.integrated_distances = config.integrated_distances
        self.use_edge_features = config.use_edge_features
        self.control_edges = config.control_edges
        if config.use_edge_features:
            d_edge = 11 if not config.integrated_distances else 12
            self.edges_feature_layer = EdgeFeaturesLayer(config.d_model, d_edge, config.h, config.dropout)

    def forward(self, query, key, value, adj_matrix, distance_matrix, edges_att, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # Prepare distances matrix
        distance_matrix = distance_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
        distance_matrix = self.distance_matrix_kernel(distance_matrix)
        p_dist = distance_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        if self.use_edge_features:
            if self.integrated_distances:
                edges_att = torch.cat((edges_att, distance_matrix.unsqueeze(1)), dim=1)
            edges_att = self.edges_feature_layer(edges_att)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn, self.self_attn = attention(query, key, value, adj_matrix,
                                                 p_dist, edges_att,
                                                 mask=mask, dropout=self.dropout,
                                                 lambdas=self.lambdas,
                                                 trainable_lambda=self.trainable_lambda,
                                                 distance_matrix_kernel=self.distance_matrix_kernel,
                                                 use_edge_features=self.use_edge_features,
                                                 control_edges=self.control_edges)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


### Conv 1x1 aka Positionwise feed forward

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = config.N_dense
        self.linears = clones(nn.Linear(config.d_model, config.d_model), config.N_dense)
        self.dropout = clones(nn.Dropout(config.dropout), config.N_dense)
        self.leaky_relu_slope = config.leaky_relu_slope
        if config.dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif config.dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif config.dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x

    def forward(self, x):
        if self.N_dense == 0:
            return x

        for i in range(len(self.linears) - 1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))

        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))


## Embeddings

class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(config.d_atom, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))
