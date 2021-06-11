"""
This implementation is adapted from
https://github.com/tencent-ailab/grover/main/grover/grover/models.py
https://github.com/tencent-ailab/grover/main/grover/grover/layers.py
"""

import math
from typing import Union, Dict, Type

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_api import PretrainedModelBase
from .models_common_utils import MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection, \
    get_activation_function
from .models_grover_utils import select_neighbor_and_aggregate
from ..configuration import GroverConfig
from ..featurization.featurization_grover import GroverBatchEncoding, GroverFeaturizer

GROVER_MODEL_ARCH = {
    'grover_base': 'https://drive.google.com/uc?id=1FbBQFRbEIItfCjumjvY_zepGq56dSClU',
    'grover_large': 'https://drive.google.com/uc?id=1eQfPPWghmaq-dvWNO-bhom1UsovKn2h5'
}


class GroverModel(PretrainedModelBase[GroverBatchEncoding, GroverConfig]):
    @classmethod
    def _get_archive_dict(cls) -> dict:
        return GROVER_MODEL_ARCH

    @classmethod
    def get_config_cls(cls) -> Type[GroverConfig]:
        return GroverConfig

    @classmethod
    def get_featurizer_cls(cls) -> Type[GroverFeaturizer]:
        return GroverFeaturizer

    def __init__(self, config: GroverConfig):
        super(GroverModel, self).__init__(config)

        self.hidden_size = config.d_model

        self.grover = GroverEncoderWrapper(config)

        if config.readout_self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=config.readout_attn_hidden,
                                   attn_out=config.readout_attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        self.mol_atom_from_atom_ffn = self.create_ffn(config)
        self.mol_atom_from_bond_ffn = self.create_ffn(config)

        self.init_weights(config.init_type)

    def create_ffn(self, config: GroverConfig):
        """
        Creates the feed-forward network for the model.
        """

        if config.ffn_features_only:
            first_linear_dim = config.features_size + config.ffn_d_generated_features
        else:
            if config.readout_self_attention:
                first_linear_dim = config.d_model * config.readout_attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += config.ffn_d_generated_features
            else:
                first_linear_dim = config.d_model + config.ffn_d_generated_features

        dropout = nn.Dropout(config.dropout)
        activation = get_activation_function(config.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if config.ffn_n_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, config.readout_n_outputs)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, config.ffn_d_hidden)
            ]
            for _ in range(config.ffn_n_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(config.ffn_d_hidden, config.ffn_d_hidden),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(config.ffn_d_hidden, config.readout_n_outputs),
            ])

        return nn.Sequential(*ffn)

    def forward(self, batch: GroverBatchEncoding):
        _, _, _, _, _, a_scope, _, _ = batch.get_components()

        output = self.grover(batch)
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if batch.generated_features is not None:
            mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, batch.generated_features], 1)
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, batch.generated_features], 1)

        atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
        bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
        return atom_ffn_output, bond_ffn_output


class GroverEncoderWrapper(nn.Module):
    """
    The GROVER Embedding class. It contains the GTransEncoder.
    This GTransEncoder can be replaced by any validate encoders.
    """

    def __init__(self, config: GroverConfig):
        super(GroverEncoderWrapper, self).__init__()
        self.embedding_output_type = config.encoder_output_type
        edge_dim = config.d_bond + config.d_atom
        node_dim = config.d_atom
        if not hasattr(config, "backbone"):
            print("No backbone specified in args, use gtrans backbone.")
            config.backbone = "gtrans"
        if config.backbone == "gtrans" or config.backbone == "dualtrans":
            # dualtrans is the old name.
            self.encoders = GroverEncoder(config,
                                          hidden_size=config.d_model,
                                          edge_fdim=edge_dim,
                                          node_fdim=node_dim,
                                          dropout=config.dropout,
                                          activation=config.activation,
                                          num_mt_block=config.encoder_n_blocks,
                                          num_attn_head=config.encoder_n_attn_heads,
                                          atom_emb_output=self.embedding_output_type,
                                          bias=config.encoder_attn_output_bias)

    def forward(self, batch: GroverBatchEncoding) -> Dict:
        """
        The forward function takes graph_batch as input and output a dict. The content of the dict is decided by
        self.embedding_output_type.
        """
        output = self.encoders(batch)
        if self.embedding_output_type == 'atom':
            return {"atom_from_atom": output[0], "atom_from_bond": output[1],
                    "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        elif self.embedding_output_type == 'bond':
            return {"atom_from_atom": None, "atom_from_bond": None,
                    "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        elif self.embedding_output_type == "both":
            return {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                    "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}


class GroverEncoder(nn.Module):
    def __init__(self,
                 config: GroverConfig,
                 hidden_size,
                 edge_fdim,
                 node_fdim,
                 dropout=0.0,
                 activation="ReLU",
                 num_mt_block=1,
                 num_attn_head=4,
                 atom_emb_output: Union[bool, str] = False,  # options: True, False, None, "atom", "bond", "both"
                 bias=False,
                 res_connection=False):
        """

        :param config: the arguments.
        :param hidden_size: the hidden size of the model.
        :param edge_fdim: the dimension of additional feature for edge/bond.
        :param node_fdim: the dimension of additional feature for node/atom.
        :param dropout: the dropout ratio
        :param activation: the activation function
        :param num_mt_block: the number of mt block.
        :param num_attn_head: the number of attention head.
        :param atom_emb_output:  enable the output aggregation after message passing.
                                              atom_messages:      True                      False
        -False: no aggregating to atom. output size:     (num_atoms, hidden_size)    (num_bonds, hidden_size)
        -True:  aggregating to atom.    output size:     (num_atoms, hidden_size)    (num_atoms, hidden_size)
        -None:                         same as False
        -"atom":                       same as True
        -"bond": aggragating to bond.   output size:     (num_bonds, hidden_size)    (num_bonds, hidden_size)
        -"both": aggregating to atom&bond. output size:  (num_atoms, hidden_size)    (num_bonds, hidden_size)
                                                         (num_bonds, hidden_size)    (num_atoms, hidden_size)
        :param bias: enable bias term in all linear layers.
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(GroverEncoder, self).__init__()

        # For the compatibility issue.
        if atom_emb_output is False:
            atom_emb_output = None
        if atom_emb_output is True:
            atom_emb_output = 'atom'

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.res_connection = res_connection
        self.edge_blocks = nn.ModuleList()
        self.node_blocks = nn.ModuleList()

        edge_input_dim = edge_fdim
        node_input_dim = node_fdim
        edge_input_dim_i = edge_input_dim
        node_input_dim_i = node_input_dim

        for i in range(num_mt_block):
            if i != 0:
                edge_input_dim_i = self.hidden_size
                node_input_dim_i = self.hidden_size
            self.edge_blocks.append(MTBlock(config=config,
                                            num_attn_head=num_attn_head,
                                            input_dim=edge_input_dim_i,
                                            hidden_size=self.hidden_size,
                                            activation=activation,
                                            dropout=dropout,
                                            bias=self.bias,
                                            atom_messages=False))
            self.node_blocks.append(MTBlock(config=config,
                                            num_attn_head=num_attn_head,
                                            input_dim=node_input_dim_i,
                                            hidden_size=self.hidden_size,
                                            activation=activation,
                                            dropout=dropout,
                                            bias=self.bias,
                                            atom_messages=True))

        self.atom_emb_output = atom_emb_output

        self.ffn_atom_from_atom = PositionwiseFeedForward(d_input=self.hidden_size + node_fdim,
                                                          d_hidden=self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_output=self.hidden_size,
                                                          n_layers=2)

        self.ffn_atom_from_bond = PositionwiseFeedForward(d_input=self.hidden_size + node_fdim,
                                                          d_hidden=self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_output=self.hidden_size,
                                                          n_layers=2)

        self.ffn_bond_from_atom = PositionwiseFeedForward(d_input=self.hidden_size + edge_fdim,
                                                          d_hidden=self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_output=self.hidden_size,
                                                          n_layers=2)

        self.ffn_bond_from_bond = PositionwiseFeedForward(d_input=self.hidden_size + edge_fdim,
                                                          d_hidden=self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_output=self.hidden_size,
                                                          n_layers=2)

        self.atom_from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.atom_from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.bond_from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.bond_from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)

        self.act_func_node = get_activation_function(self.activation)
        self.act_func_edge = get_activation_function(self.activation)

        self.dropout_layer = nn.Dropout(p=config.dropout)

    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for atom view.
        aggregate to atom.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param atom_fea: the atom/node feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def pointwise_feed_forward_to_bond_embedding(self, emb_output, bond_fea, a2nei, b2revb, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for bond view.
        aggregate to bond.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param bond_fea: the bond/edge feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, a2nei)
        # remove rev bond / atom --- need for bond view
        aggr_output = self.remove_rev_bond_message(emb_output, aggr_output, b2revb)
        aggr_outputx = torch.cat([bond_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    @staticmethod
    def remove_rev_bond_message(orginal_message, aggr_message, b2revb):
        """

        :param orginal_message:
        :param aggr_message:
        :param b2revb:
        :return:
        """
        rev_message = orginal_message[b2revb]
        return aggr_message - rev_message

    def atom_bond_transform(self,
                            to_atom=True,  # False: to bond
                            atomwise_input=None,
                            bondwise_input=None,
                            original_f_atoms=None,
                            original_f_bonds=None,
                            a2a=None,
                            a2b=None,
                            b2a=None,
                            b2revb=None
                            ):
        """
        Transfer the output of atom/bond multi-head attention to the final atom/bond output.
        :param to_atom: if true, the output is atom emebedding, otherwise, the output is bond embedding.
        :param atomwise_input: the input embedding of atom/node.
        :param bondwise_input: the input embedding of bond/edge.
        :param original_f_atoms: the initial atom features.
        :param original_f_bonds: the initial bond features.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param a2b: mapping from atom index to incoming bond indices.
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """

        if to_atom:
            # atom input to atom output
            atomwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(atomwise_input, original_f_atoms, a2a,
                                                                              self.ffn_atom_from_atom)
            atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)
            # bond to atom
            bondwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(bondwise_input, original_f_atoms, a2b,
                                                                              self.ffn_atom_from_bond)
            bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)
            return atom_in_atom_out, bond_in_atom_out
        else:  # to bond embeddings

            # atom input to bond output
            atom_list_for_bond = torch.cat([b2a.unsqueeze(dim=1), a2a[b2a]], dim=1)
            atomwise_input, _ = self.pointwise_feed_forward_to_bond_embedding(atomwise_input, original_f_bonds,
                                                                              atom_list_for_bond,
                                                                              b2a[b2revb], self.ffn_bond_from_atom)
            atom_in_bond_out = self.bond_from_atom_sublayer(None, atomwise_input)
            # bond input to bond output
            bond_list_for_bond = a2b[b2a]
            bondwise_input, _ = self.pointwise_feed_forward_to_bond_embedding(bondwise_input, original_f_bonds,
                                                                              bond_list_for_bond,
                                                                              b2revb, self.ffn_bond_from_bond)
            bond_in_bond_out = self.bond_from_bond_sublayer(None, bondwise_input)
            return atom_in_bond_out, bond_in_bond_out

    def forward(self, batch: GroverBatchEncoding, features_batch=None):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch.get_components()

        node_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        edge_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a

        # opt pointwise_feed_forward
        original_f_atoms, original_f_bonds = f_atoms, f_bonds

        # Note: features_batch is not used here.
        for nb in self.node_blocks:  # atom messages. Multi-headed attention
            node_batch, features_batch = nb(node_batch, features_batch)
        for eb in self.edge_blocks:  # bond messages. Multi-headed attention
            edge_batch, features_batch = eb(edge_batch, features_batch)

        atom_output, _, _, _, _, _, _, _ = node_batch  # atom hidden states
        _, bond_output, _, _, _, _, _, _ = edge_batch  # bond hidden states

        if self.atom_emb_output is None:
            # output the embedding from multi-head attention directly.
            return atom_output, bond_output

        if self.atom_emb_output == 'atom':
            return self.atom_bond_transform(to_atom=True,  # False: to bond
                                            atomwise_input=atom_output,
                                            bondwise_input=bond_output,
                                            original_f_atoms=original_f_atoms,
                                            original_f_bonds=original_f_bonds,
                                            a2a=a2a,
                                            a2b=a2b,
                                            b2a=b2a,
                                            b2revb=b2revb)
        elif self.atom_emb_output == 'bond':
            return self.atom_bond_transform(to_atom=False,  # False: to bond
                                            atomwise_input=atom_output,
                                            bondwise_input=bond_output,
                                            original_f_atoms=original_f_atoms,
                                            original_f_bonds=original_f_bonds,
                                            a2a=a2a,
                                            a2b=a2b,
                                            b2a=b2a,
                                            b2revb=b2revb)
        else:  # 'both'
            atom_embeddings = self.atom_bond_transform(to_atom=True,  # False: to bond
                                                       atomwise_input=atom_output,
                                                       bondwise_input=bond_output,
                                                       original_f_atoms=original_f_atoms,
                                                       original_f_bonds=original_f_bonds,
                                                       a2a=a2a,
                                                       a2b=a2b,
                                                       b2a=b2a,
                                                       b2revb=b2revb)

            bond_embeddings = self.atom_bond_transform(to_atom=False,  # False: to bond
                                                       atomwise_input=atom_output,
                                                       bondwise_input=bond_output,
                                                       original_f_atoms=original_f_atoms,
                                                       original_f_bonds=original_f_bonds,
                                                       a2a=a2a,
                                                       a2b=a2b,
                                                       b2a=b2a,
                                                       b2revb=b2revb)
            # Notice: need to be consistent with output format of DualMPNN encoder
            return ((atom_embeddings[0], bond_embeddings[0]),
                    (atom_embeddings[1], bond_embeddings[1]))


# Message Passing

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, config: GroverConfig,
                 atom_messages: bool,
                 init_message_dim: int,
                 attached_fea_fdim: int,
                 hidden_size: int,
                 bias: bool,
                 depth: int,
                 dropout: float,
                 undirected: bool,
                 dense: bool,
                 aggregate_to_atom: bool,
                 attach_fea: bool,
                 input_layer="fc",
                 dynamic_depth='none'
                 ):
        """
        Initializes the MPNEncoder.
        :param config: the arguments.
        :param atom_messages: enables atom_messages or not.
        :param init_message_dim:  the initial input message dimension.
        :param attached_fea_fdim:  the attached feature dimension.
        :param hidden_size: the output message dimension during message passing.
        :param bias: the bias in the message passing.
        :param depth: the message passing depth.
        :param dropout: the dropout rate.
        :param undirected: the message passing is undirected or not.
        :param dense: enables the dense connections.
        :param attach_fea: enables the feature attachment during the message passing process.
        :param dynamic_depth: enables the dynamic depth. Possible choices: "none", "uniform" and "truncnorm"
        """
        super(MPNEncoder, self).__init__()
        self.init_message_dim = init_message_dim
        self.attached_fea_fdim = attached_fea_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.input_layer = input_layer
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.dense = dense
        self.aggreate_to_atom = aggregate_to_atom
        self.attached_fea = attach_fea
        self.dynamic_depth = dynamic_depth

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(config.activation)

        # Input
        if self.input_layer == "fc":
            input_dim = self.init_message_dim
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.attached_fea:
            w_h_input_size = self.hidden_size + self.attached_fea_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

    def forward(self,
                init_messages,
                init_attached_features,
                a2nei,
                a2attached,
                b2a=None,
                b2revb=None,
                adjs=None
                ) -> torch.FloatTensor:
        """
        The forward function.
        :param init_messages:  initial massages, can be atom features or bond features.
        :param init_attached_features: initial attached_features.
        :param a2nei: the relation of item to its neighbors. For the atom message passing, a2nei = a2a. For bond
        messages a2nei = a2b
        :param a2attached: the relation of item to the attached features during message passing. For the atom message
        passing, a2attached = a2b. For the bond message passing a2attached = a2a
        :param b2a: remove the reversed bond in bond message passing
        :param b2revb: remove the revered atom in bond message passing
        :return: if aggreate_to_atom or self.atom_messages, return num_atoms x hidden.
        Otherwise, return num_bonds x hidden
        """

        # Input
        if self.input_layer == 'fc':
            input = self.W_i(init_messages)  # num_bonds x hidden_size # f_bond
            message = self.act_func(input)  # num_bonds x hidden_size
        elif self.input_layer == 'none':
            input = init_messages
            message = input

        attached_fea = init_attached_features  # f_atom / f_bond

        # dynamic depth
        # uniform sampling from depth - 1 to depth + 1
        # only works in training.
        if self.training and self.dynamic_depth != "none":
            if self.dynamic_depth == "uniform":
                # uniform sampling
                ndepth = np.random.randint(self.depth - 3, self.depth + 3)
            else:
                # truncnorm
                mu = self.depth
                sigma = 1
                lower = mu - 3 * sigma
                upper = mu + 3 * sigma
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                ndepth = int(X.rvs(1))
        else:
            ndepth = self.depth

        # Message passing
        for _ in range(ndepth - 1):
            if self.undirected:
                # two directions should be the same
                message = (message + message[b2revb]) / 2

            nei_message = select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_fea:
                attached_nei_fea = select_neighbor_and_aggregate(attached_fea, a2attached)
                a_message = torch.cat((nei_message, attached_nei_fea), dim=1)

            if not self.atom_messages:
                rev_message = message[b2revb]
                if self.attached_fea:
                    atom_rev_message = attached_fea[b2a[b2revb]]
                    rev_message = torch.cat((rev_message, atom_rev_message), dim=1)
                # Except reverse bond its-self(w) ! \sum_{k\in N(u) \ w}
                message = a_message[b2a] - rev_message  # num_bonds x hidden
            else:
                message = a_message

            message = self.W_h(message)

            # BUG here, by default MPNEncoder use the dense connection in the message passing step.
            # The correct form should if not self.dense
            if self.dense:
                message = self.act_func(message)  # num_bonds x hidden_size
            else:
                message = self.act_func(input + message)
            message = self.dropout_layer(message)  # num_bonds x hidden

        output = message

        return output  # num_atoms x hidden


# Attention

class MTBlock(nn.Module):
    """
    The Multi-headed attention block.
    """

    def __init__(self,
                 config: GroverConfig,
                 num_attn_head,
                 input_dim,
                 hidden_size,
                 activation="ReLU",
                 dropout=0.0,
                 bias=True,
                 atom_messages=False,
                 res_connection=False):
        """

        :param config: the arguments.
        :param num_attn_head: the number of attention head.
        :param input_dim: the input dimension.
        :param hidden_size: the hidden size of the model.
        :param activation: the activation function.
        :param dropout: the dropout ratio
        :param bias: if true: all linear layer contains bias term.
        :param atom_messages: the MPNEncoder type
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(MTBlock, self).__init__()
        # self.args = args
        self.atom_messages = atom_messages
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList()
        self.input_dim = input_dim
        self.res_connection = res_connection
        self.act_func = get_activation_function(activation)
        self.dropout_layer = nn.Dropout(p=dropout)
        # Note: elementwise_affine has to be consistent with the pre-training phase
        self.layernorm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

        self.W_i = nn.Linear(self.input_dim, self.hidden_size, bias=bias)
        self.attn = MultiHeadedAttention(h=num_attn_head,
                                         d_model=self.hidden_size,
                                         output_bias=bias,
                                         dropout=dropout,
                                         attention=GroverAttention())
        self.W_o = nn.Linear(self.hidden_size * num_attn_head, self.hidden_size, bias=bias)
        self.sublayer = SublayerConnection(size=self.hidden_size, dropout=dropout)
        for _ in range(num_attn_head):
            self.heads.append(Head(config, hidden_size=hidden_size, atom_messages=atom_messages))

    def forward(self, batch, features_batch=None):
        """

        :param batch: the graph batch generated by GroverCollator.
        :param features_batch: the additional features of molecules. (deprecated)
        :return:
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

        if self.atom_messages:
            # Only add linear transformation in the input feature.
            if f_atoms.shape[1] != self.hidden_size:
                f_atoms = self.W_i(f_atoms)
                f_atoms = self.dropout_layer(self.layernorm(self.act_func(f_atoms)))

        else:  # bond messages
            if f_bonds.shape[1] != self.hidden_size:
                f_bonds = self.W_i(f_bonds)
                f_bonds = self.dropout_layer(self.layernorm(self.act_func(f_bonds)))

        queries = []
        keys = []
        values = []
        for head in self.heads:
            q, k, v = head(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
            queries.append(q.unsqueeze(1))
            keys.append(k.unsqueeze(1))
            values.append(v.unsqueeze(1))
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)

        x_out = self.attn(queries, keys, values)  # multi-headed attention
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.W_o(x_out)

        x_in = None
        # support no residual connection in MTBlock.
        if self.res_connection:
            if self.atom_messages:
                x_in = f_atoms
            else:
                x_in = f_bonds

        if self.atom_messages:
            f_atoms = self.sublayer(x_in, x_out)
        else:
            f_bonds = self.sublayer(x_in, x_out)

        batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        features_batch = features_batch
        return batch, features_batch


class GroverAttention(nn.Module):
    """
    Compute 'Scaled Dot Product SelfAttention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :param dropout:
        :return:
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class Head(nn.Module):
    """
    One head for multi-headed attention.
    :return: (query, key, value)
    """

    def __init__(self, config: GroverConfig, hidden_size, atom_messages=False):
        """
        Initialization.
        :param config: The argument.
        :param hidden_size: the dimension of hidden layer in Head.
        :param atom_messages: the MPNEncoder type.
        """
        super(Head, self).__init__()
        atom_fdim = hidden_size
        bond_fdim = hidden_size
        hidden_size = hidden_size
        self.atom_messages = atom_messages
        if self.atom_messages:
            init_message_dim = atom_fdim
            attached_fea_dim = bond_fdim
        else:
            init_message_dim = bond_fdim
            attached_fea_dim = atom_fdim

        # Here we use the message passing network as query, key and value.
        self.mpn_q = MPNEncoder(config=config,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=config.encoder_attn_output_bias,
                                depth=config.mpn_depth,
                                dropout=config.dropout,
                                undirected=config.mpn_undirected,
                                dense=config.mpn_dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")
        self.mpn_k = MPNEncoder(config=config,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=config.encoder_attn_output_bias,
                                depth=config.mpn_depth,
                                dropout=config.dropout,
                                undirected=config.mpn_undirected,
                                dense=config.mpn_dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")
        self.mpn_v = MPNEncoder(config=config,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=config.encoder_attn_output_bias,
                                depth=config.mpn_depth,
                                dropout=config.dropout,
                                undirected=config.mpn_undirected,
                                dense=config.mpn_dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")

    def forward(self, f_atoms, f_bonds, a2b, a2a, b2a, b2revb):
        """
        The forward function.
        :param f_atoms: the atom features, num_atoms * atom_dim
        :param f_bonds: the bond features, num_bonds * bond_dim
        :param a2b: mapping from atom index to incoming bond indices.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """
        if self.atom_messages:
            init_messages = f_atoms
            init_attached_features = f_bonds
            a2nei = a2a
            a2attached = a2b
            b2a = b2a
            b2revb = b2revb
        else:
            init_messages = f_bonds
            init_attached_features = f_atoms
            a2nei = a2b
            a2attached = a2a
            b2a = b2a
            b2revb = b2revb

        q = self.mpn_q(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        k = self.mpn_k(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        v = self.mpn_v(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        return q, k, v


# Readout

class Readout(nn.Module):
    """The readout function. Convert the node embeddings to the graph embeddings."""

    def __init__(self,
                 rtype: str = "none",
                 hidden_size: int = 0,
                 attn_hidden: int = None,
                 attn_out: int = None,
                 ):
        """
        The readout function.
        :param rtype: readout type, can be "mean" and "self_attention".
        :param hidden_size: input hidden size
        :param attn_hidden: only valid if rtype == "self_attention". The attention hidden size.
        :param attn_out: only valid if rtype == "self_attention". The attention out size.
        :param args: legacy use.
        """
        super(Readout, self).__init__()
        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.rtype = "mean"

        if rtype == "self_attention":
            self.attn = ReadoutSelfAttention(hidden=attn_hidden,
                                             in_feature=hidden_size,
                                             out_feature=attn_out)
            self.rtype = "self_attention"

    def forward(self, embeddings, scope):
        """
        The forward function, given a batch node/edge embedding and a scope list,
        produce the graph-level embedding by a scope.
        :param embeddings: The embedding matrix, num_atoms or num_bonds \times hidden_size.
        :param scope: a list, in which the element is a list [start, range]. `start` is the index
        :return:
        """
        # Readout
        mol_vecs = []
        self.attns = []
        for _, (a_start, a_size) in enumerate(scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = embeddings.narrow(0, a_start, a_size)
                if self.rtype == "self_attention":
                    cur_hiddens, attn = self.attn(cur_hiddens)
                    cur_hiddens = cur_hiddens.flatten()
                    # Temporarily disable. Enable it if you want to save attentions.
                    # self.attns.append(attn.cpu().detach().numpy())
                else:
                    cur_hiddens = cur_hiddens.sum(dim=0) / a_size
                mol_vecs.append(cur_hiddens)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        return mol_vecs


class ReadoutSelfAttention(nn.Module):
    """
       Self SelfAttention Layer
       Given $X\in \mathbb{R}^{n \times in_feature}$, the attention is calculated by: $a=Softmax(W_2tanh(W_1X))$, where
       $W_1 \in \mathbb{R}^{hidden \times in_feature}$, $W_2 \in \mathbb{R}^{out_feature \times hidden}$.
       The final output is: $out=aX$, which is unrelated with input $n$.
    """

    def __init__(self, *, hidden, in_feature, out_feature):
        """
        The init function.
        :param hidden: the hidden dimension, can be viewed as the number of experts.
        :param in_feature: the input feature dimension.
        :param out_feature: the output feature dimension.
        """
        super(ReadoutSelfAttention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(out_feature, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X):
        """
        The forward function.
        :param X: The input feature map. $X \in \mathbb{R}^{n \times in_feature}$.
        :return: The final embeddings and attention matrix.
        """
        x = torch.tanh(torch.matmul(self.w1, X.transpose(1, 0)))
        x = torch.matmul(self.w2, x)
        attn = torch.nn.functional.softmax(x, dim=-1)
        x = torch.matmul(attn, X)
        return x, attn
