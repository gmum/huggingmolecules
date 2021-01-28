"""
This implementation is adapted from
https://github.com/ardigen/MAT/blob/master/src/transformer.py
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_api import PretrainedModelBase
from .. import PatConfig
from ..featurization.featurization_pat import PatBatchEncoding, PatFeaturizer

PAT_PRETRAINED_NAME_TO_WEIGHTS_ARCH_MAPPING = {
}


class PatModel(PretrainedModelBase[PatBatchEncoding, PatConfig]):

    def __init__(self, config: PatConfig):
        super(PatModel, self).__init__(config)
        c = copy.deepcopy
        attn = MultiHeadedAttention(config.h, config.d_model, config.edge_dim, config.dropout)
        ff = PositionwiseFeedForward(config.d_model, config.N_dense, config.dropout)
        self.encoder = Encoder(EncoderLayer(config.d_model, c(attn), c(ff), config.dropout), config.N)
        self.src_embed = Embeddings(config.d_model, config.d_atom, config.dropout)
        self.generator = Generator(config.d_model, config.aggregation_type, config.n_output, config.n_generator_layers,
                                   config.dropout)
        self.dist_rbf = BesselBasisLayerEnvelope(num_radial=config.num_radial, cutoff=config.cutoff)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        pass

    @classmethod
    def get_config_cls(cls):
        return PatConfig

    @classmethod
    def get_featurizer_cls(cls):
        return PatFeaturizer

    def forward(self, batch: PatBatchEncoding):
        "Take in and process masked src and target sequences."
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        distances_matrix = self.dist_rbf(batch.distance_matrix)
        edges_att = torch.cat((batch.bond_features, distances_matrix), dim=1)
        return self.predict(self.encode(batch.node_features, batch_mask, edges_att), batch_mask)

    def encode(self, src, src_mask, edges_att):
        return self.encoder(self.src_embed(src), src_mask, edges_att)

    def predict(self, out, out_mask):
        return self.generator(out, out_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

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
                self.proj.append(LayerNorm(attn_hidden))
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


# Encoder

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, edges_att):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, edges_att)
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


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, edges_att):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, edges_att, mask))
        return self.sublayer[1](x, self.feed_forward)


# Attention

class EdgeFeaturesLayer(nn.Module):
    def __init__(self, d_edge, d_out, d_hidden, h, dropout):
        super(EdgeFeaturesLayer, self).__init__()
        self.d_k = d_out // h
        self.h = h

        self.nn = nn.Sequential(
            nn.Linear(d_edge, d_hidden),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, p_edge):
        p_edge = p_edge.permute(0, 2, 3, 1)
        p_edge = self.nn(p_edge).permute(0, 3, 1, 2)
        p_edge = p_edge.view(p_edge.size(0), self.h, self.d_k, p_edge.size(2), p_edge.size(3))
        return p_edge


def attention(query, key, value,
              relative_K, relative_V,
              relative_u, relative_v,
              mask=None, dropout=None, eps=1e-6, inf=1e12):
    "Compute 'Scaled Dot Product Attention'"
    #     d_k = query.size(-1)
    b, h, n, d_k = query.size(0), query.size(1), query.size(2), query.size(-1)

    scores1 = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #     scores2 = torch.matmul(query.view(b, h, n, 1, d_k), relative_K.permute(0, 1, 3, 2, 4)).view(b, h, n, n) / math.sqrt(d_k)
    #     scores3 = torch.matmul(key.view(b, h, n, 1, d_k), relative_K.permute(0, 1, 3, 2, 4)).view(b, h, n, n) / math.sqrt(d_k)
    scores2 = torch.matmul((query + key).view(b, h, n, 1, d_k), relative_K.permute(0, 1, 3, 2, 4)).view(b, h, n,
                                                                                                        n) / math.sqrt(
        d_k)
    scores3 = torch.matmul(key, relative_u.transpose(-2, -1))
    scores4 = torch.matmul(relative_K.permute(0, 1, 3, 4, 2), relative_v).squeeze(-1)
    scores = scores1 + scores2 + scores3 + scores4  # + scores5

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0, -inf)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    atoms_features1 = torch.matmul(p_attn, value)
    atoms_features2 = (p_attn.unsqueeze(2) * relative_V).sum(-1).permute(0, 1, 3, 2)

    atoms_features = atoms_features1 + atoms_features2

    return atoms_features, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, edge_dim, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.relative_K = EdgeFeaturesLayer(edge_dim, d_model, self.d_k, h, dropout)
        self.relative_V = EdgeFeaturesLayer(edge_dim, d_model, self.d_k, h, dropout)

        self.relative_u = nn.Parameter(torch.empty(1, self.h, 1, self.d_k))
        self.relative_v = nn.Parameter(torch.empty(1, self.h, 1, self.d_k, 1))

    def forward(self, query, key, value, edges_att, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        relative_K = self.relative_K(edges_att)
        relative_V = self.relative_V(edges_att)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 relative_K, relative_V,
                                 self.relative_u, self.relative_v,
                                 mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# Conv 1x1 aka Positionwise feed forward

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, N_dense, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        lin_factor = 2
        if N_dense == 1:
            self.linears = [nn.Linear(d_model, d_model)]
        else:
            #             self.linears = [nn.Linear(d_model, d_model*4)] + [nn.Linear(d_model*4, d_model*4) for _ in range(N_dense-2)] + [nn.Linear(d_model*4, d_model)]
            self.linears = [nn.Linear(d_model, d_model * lin_factor)] + [
                nn.Linear(d_model * lin_factor, d_model * lin_factor) for _ in range(N_dense - 2)] + [
                               nn.Linear(d_model * lin_factor, d_model)]

        self.linears = nn.ModuleList(self.linears)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.nonlinearity = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        if self.N_dense == 0:
            return x

        for i in range(self.N_dense - 1):
            x = self.dropout[i](self.nonlinearity(self.linears[i](x)))

        return self.linears[-1](x)


# Embeddings

class Embeddings(nn.Module):
    def __init__(self, d_model, d_atom, dropout):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))


# Distance Layers

class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    """

    def __init__(self, exponent, **kwargs):
        super().__init__(**kwargs)
        self.exponent = exponent

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs):
        # Envelope function divided by r
        env_val = 1 / inputs + self.a * inputs ** (self.p - 1) + self.b * inputs ** self.p + self.c * inputs ** (
                self.p + 1)

        return torch.where(inputs < 1, env_val, torch.zeros_like(inputs))


class BesselBasisLayerEnvelope(nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent=5, **kwargs):
        super().__init__(**kwargs)
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.sqrt_cutoff = np.sqrt(2. / cutoff)
        self.inv_cutoff = 1. / cutoff
        self.envelope = Envelope(envelope_exponent)

        self.frequencies = np.pi * torch.arange(1, num_radial + 1).float()

    def forward(self, inputs):
        inputs = inputs.unsqueeze(-1) + 1e-6
        d_scaled = inputs * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        return (d_cutoff * torch.sin(self.frequencies * d_scaled)).permute(0, 3, 1, 2)
