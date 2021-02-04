"""
This implementation is adapted from
https://github.com/ardigen/MAT/blob/master/src/transformer.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_api import PretrainedModelBase
from .models_common_utils import clones
from .models_mat import Embeddings, Generator, PositionwiseFeedForward, Encoder
from .. import PatConfig
from ..featurization.featurization_pat import PatBatchEncoding, PatFeaturizer

PAT_PRETRAINED_NAME_TO_WEIGHTS_ARCH_MAPPING = {
    'pat_test': './pretrained/pat/weights/pat_test.pt',
}


class PatModel(PretrainedModelBase[PatBatchEncoding, PatConfig]):
    @classmethod
    def get_config_cls(cls):
        return PatConfig

    @classmethod
    def get_featurizer_cls(cls):
        return PatFeaturizer

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        return PAT_PRETRAINED_NAME_TO_WEIGHTS_ARCH_MAPPING.get(pretrained_name, None)

    def __init__(self, config: PatConfig):
        super().__init__(config)

        self.src_embed = Embeddings(config.d_atom, config.d_model, config.dropout)
        self.dist_rbf = BesselBasisLayerEnvelope(num_radial=config.num_radial, cutoff=config.cutoff)
        attn = MultiHeadedAttention(config.h, config.d_model, config.edge_dim, config.dropout)
        ff = PositionwiseFeedForward(config.d_model, config.N_dense, config.dropout, config.lin_factor)
        self.encoder = Encoder(attn, ff, config.d_model, config.dropout, config.N)
        self.generator = Generator(config.d_model, config.aggregation_type, config.n_output, config.n_generator_layers,
                                   config.dropout)

        self.init_weights(config.init_type)

    def forward(self, batch: PatBatchEncoding):
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = self.src_embed(batch.node_features)
        distances_matrix = self.dist_rbf(batch.distance_matrix)
        edges_att = torch.cat((batch.bond_features, distances_matrix), dim=1)
        encoded = self.encoder(embedded, batch_mask, edges_att=edges_att)
        output = self.generator(encoded, batch_mask)
        return output


# Distance Layers

class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    """

    def __init__(self, exponent):
        super().__init__()
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
        super().__init__()
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


# Attention

class PatEdgeFeaturesLayer(nn.Module):
    def __init__(self, d_edge, d_out, d_hidden, h, dropout):
        super(PatEdgeFeaturesLayer, self).__init__()
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


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, edge_dim, dropout=0.1):
        assert d_model % h == 0
        super(MultiHeadedAttention, self).__init__()

        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

        self.relative_K = PatEdgeFeaturesLayer(edge_dim, d_model, self.d_k, h, dropout)
        self.relative_V = PatEdgeFeaturesLayer(edge_dim, d_model, self.d_k, h, dropout)

        self.relative_u = nn.Parameter(torch.empty(1, self.h, 1, self.d_k))
        self.relative_v = nn.Parameter(torch.empty(1, self.h, 1, self.d_k, 1))

        self.attn = None

    def forward(self, query, key, value, edges_att, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batches = query.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value,
                                      edges_att, mask)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self,
                  query, key, value,
                  edges_att, mask,
                  inf=1e12):
        """Compute 'Scaled Dot Product Attention'"""
        b, h, n, d_k = query.size(0), query.size(1), query.size(2), query.size(-1)

        # Prepare relative matrices
        relative_K = self.relative_K(edges_att)
        relative_V = self.relative_V(edges_att)

        scores1 = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores2 = torch.matmul((query + key).view(b, h, n, 1, d_k),
                               relative_K.permute(0, 1, 3, 2, 4)).view(b, h, n, n) / math.sqrt(d_k)
        scores3 = torch.matmul(key, self.relative_u.transpose(-2, -1))
        scores4 = torch.matmul(relative_K.permute(0, 1, 3, 4, 2), self.relative_v).squeeze(-1)
        scores = scores1 + scores2 + scores3 + scores4  # + scores5

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, h, n, 1) == 0, -inf)
        p_attn = F.softmax(scores, dim=-1)

        p_attn = self.dropout(p_attn)

        atoms_features1 = torch.matmul(p_attn, value)
        atoms_features2 = (p_attn.unsqueeze(2) * relative_V).sum(-1).permute(0, 1, 3, 2)

        atoms_features = atoms_features1 + atoms_features2

        return atoms_features, p_attn
