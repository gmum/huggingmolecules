"""
This implementation is adapted from
https://github.com/ardigen/MAT/blob/master/src/transformer.py
"""

import math
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_api import PretrainedModelBase
from .models_common_utils import PositionwiseFeedForward, MultiHeadedAttention, Embedding, Encoder, Generator
from ..configuration import MatppConfig
from ..featurization.featurization_matpp import MatppBatchEncoding, MatppFeaturizer

MATPP_MODEL_ARCH = {
    'matpp_test': 'https://drive.google.com/uc?id=1hQzufXRDG9tp9FolRbOrvFxRyuLiQ9ZA'
}


class MatppModel(PretrainedModelBase[MatppBatchEncoding, MatppConfig]):
    @classmethod
    def get_config_cls(cls) -> Type[MatppConfig]:
        return MatppConfig

    @classmethod
    def get_featurizer_cls(cls) -> Type[MatppFeaturizer]:
        return MatppFeaturizer

    @classmethod
    def _get_archive_dict(cls) -> dict:
        return MATPP_MODEL_ARCH

    def __init__(self, config: MatppConfig):
        super().__init__(config)

        # Embedding
        self.src_embed = Embedding(d_input=config.d_atom,
                                   d_output=config.d_model,
                                   dropout=config.dropout)

        # Distance
        self.dist_rbf = BesselBasisLayerEnvelope(num_radial=config.envelope_num_radial,
                                                 cutoff=config.envelope_cutoff,
                                                 exponent=config.envelope_exponent)

        # Encoder
        attention = MatppAttention(config)
        sa_layer = MultiHeadedAttention(h=config.encoder_n_attn_heads,
                                        d_model=config.d_model,
                                        dropout=config.dropout,
                                        attention=attention)
        ff_layer = PositionwiseFeedForward(d_input=config.d_model,
                                           d_hidden=config.ffn_d_hidden,
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
                                   d_output=config.generator_d_outputs,
                                   n_layers=config.generator_n_layers,
                                   dropout=config.dropout)

        # Initialization
        self.init_weights(config.init_type)

    def forward(self, batch: MatppBatchEncoding):
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = self.src_embed(batch.node_features)
        distances_matrix = self.dist_rbf(batch.distance_matrix)
        edges_att = torch.cat((batch.bond_features, distances_matrix), dim=1)
        encoded = self.encoder(embedded, batch_mask, edges_att=edges_att)
        output = self.generator(encoded, batch_mask)
        return output


# Distance Layers


class BesselBasisLayerEnvelope(nn.Module):
    def __init__(self, *,
                 num_radial: float,
                 cutoff: float,
                 exponent: float):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.sqrt_cutoff = np.sqrt(2. / cutoff)
        self.inv_cutoff = 1. / cutoff
        self.envelope = Envelope(exponent=exponent)
        self.num_radial = num_radial

    def forward(self, inputs):
        inputs = inputs.unsqueeze(-1) + 1e-6
        d_scaled = inputs * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        frequencies = np.pi * torch.arange(1, self.num_radial + 1, device=inputs.device).float()
        return (d_cutoff * torch.sin(frequencies * d_scaled)).permute(0, 3, 1, 2)


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    """

    def __init__(self, *, exponent: float):
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


# Attention

class MatppAttention(nn.Module):
    def __init__(self, config: MatppConfig):
        super().__init__()
        d_k = config.d_model // config.encoder_n_attn_heads

        self.relative_K = MatppEdgeFeaturesLayer(config)
        self.relative_V = MatppEdgeFeaturesLayer(config)

        self.relative_u = nn.Parameter(torch.empty(1, config.encoder_n_attn_heads, 1, d_k))
        self.relative_v = nn.Parameter(torch.empty(1, config.encoder_n_attn_heads, 1, d_k, 1))

    def forward(self, query, key, value, mask, dropout,
                edges_att,
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

        p_attn = dropout(p_attn)

        atoms_features1 = torch.matmul(p_attn, value)
        atoms_features2 = (p_attn.unsqueeze(2) * relative_V).sum(-1).permute(0, 1, 3, 2)

        atoms_features = atoms_features1 + atoms_features2

        return atoms_features, p_attn


class MatppEdgeFeaturesLayer(nn.Module):
    def __init__(self, config: MatppConfig):
        super(MatppEdgeFeaturesLayer, self).__init__()
        self.d_k = config.d_model // config.encoder_n_attn_heads
        self.h = config.encoder_n_attn_heads

        input_dim = config.d_edge
        hidden_dim = self.d_k
        output_dim = config.d_model

        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, p_edge):
        p_edge = p_edge.permute(0, 2, 3, 1)
        p_edge = self.nn(p_edge).permute(0, 3, 1, 2)
        p_edge = p_edge.view(p_edge.size(0), self.h, self.d_k, p_edge.size(2), p_edge.size(3))
        return p_edge