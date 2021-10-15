import copy

import torch
from torch import nn as nn
from torch.nn import functional as F


def clones(module: nn.Module, n: int):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# Embeddings

class Embedding(nn.Module):
    def __init__(self, *,
                 d_input: int,
                 d_output: int,
                 dropout: float):
        super(Embedding, self).__init__()
        self.lut = nn.Linear(d_input, d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))


# Encoder

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, *,
                 sa_layer: nn.Module,
                 ff_layer: nn.Module,
                 d_model: int,
                 dropout: float,
                 n_layers: int):
        super(Encoder, self).__init__()
        layer = EncoderLayer(sa_layer=sa_layer, ff_layer=ff_layer, d_model=d_model, dropout=dropout)
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask, **kwargs):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask, **kwargs)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, *,
                 sa_layer: nn.Module,
                 ff_layer: nn.Module,
                 d_model: int,
                 dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = sa_layer
        self.feed_forward = ff_layer
        self.sublayer = clones(LambdaSublayerConnection(size=d_model, dropout=dropout), 2)
        self.size = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask, **kwargs))
        return self.sublayer[1](x, self.feed_forward)


class LambdaSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, *,
                 size: int,
                 dropout: float):
        super(LambdaSublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, *,
                 size: int,
                 dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, outputs: torch.Tensor):
        """Apply residual connection to any sublayer with the same size."""
        if x is None:
            return self.dropout(self.norm(outputs))
        return x + self.dropout(self.norm(outputs))


# Attention

class MultiHeadedAttention(nn.Module):
    def __init__(self, *,
                 attention: nn.Module,
                 h: int,
                 d_model: int,
                 dropout: float,
                 output_bias: bool = True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # number of heads

        self.linear_layers = clones(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, d_model, output_bias)
        self.attention = attention

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None,
                **kwargs):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for layer, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout, **kwargs)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


# Feed Forward

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, *,
                 d_input: int,
                 d_hidden: int = None,
                 d_output: int = None,
                 activation: str,
                 n_layers: int,
                 dropout: float):
        super(PositionwiseFeedForward, self).__init__()
        self.n_layers = n_layers
        d_output = d_output if d_output is not None else d_input
        d_hidden = d_hidden if d_hidden is not None else d_input
        if n_layers == 1:
            self.linears = [nn.Linear(d_input, d_output)]
        else:
            self.linears = [nn.Linear(d_input, d_hidden)] + \
                           [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 2)] + \
                           [nn.Linear(d_hidden, d_output)]

        self.linears = nn.ModuleList(self.linears)
        self.dropout = clones(nn.Dropout(dropout), n_layers)
        self.act_func = get_activation_function(activation)

    def forward(self, x):
        if self.n_layers == 0:
            return x
        elif self.n_layers == 1:
            return self.dropout[0](self.act_func(self.linears[0](x)))
        else:
            for i in range(self.n_layers - 1):
                x = self.dropout[i](self.act_func(self.linears[i](x)))
            return self.linears[-1](x)


# Generator

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, *,
                 d_model: int,
                 d_generated_features: int,
                 aggregation_type: str,
                 d_output: int,
                 n_layers: int,
                 dropout: float,
                 attn_hidden: int = 128,
                 attn_out: int = 4):
        super(Generator, self).__init__()
        if aggregation_type == 'grover':
            self.att_net = nn.Sequential(
                nn.Linear(d_model, attn_hidden, bias=False),
                nn.Tanh(),
                nn.Linear(attn_hidden, attn_out, bias=False),
            )
            d_model *= attn_out

        d_model += d_generated_features
        if n_layers == 1:
            self.proj = nn.Linear(d_model, d_output)
        else:
            self.proj = []
            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(d_model, attn_hidden))
                self.proj.append(nn.LeakyReLU(negative_slope=0.1))
                self.proj.append(nn.LayerNorm(attn_hidden))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(attn_hidden, d_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type

    def forward(self, x, mask, generated_features):
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
        if generated_features is not None:
            out_avg_pooling = torch.cat([out_avg_pooling, generated_features], 1)
        projected = self.proj(out_avg_pooling)
        return projected


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')
