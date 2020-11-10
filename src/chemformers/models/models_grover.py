import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv, MessagePassing
from torch_geometric.utils import add_self_loops


class Grover(nn.Module):
    def __init__(self,
                 model_dim: int, num_layers: int, num_heads: int, dropout: float,
                 num_layers_dympnn: int, num_hops_dympnn: int,
                 n_f_atom: int, n_f_bond: int,
                 readout_hidden_dim: int, readout_num_heads: int,
                 head_hidden_dim: int, output_dim: int, head_num_layers: int):
        super(Grover, self).__init__()

        self.enc = Encoder(model_dim=model_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                           num_layers_dympnn=num_layers_dympnn, num_hops_dympnn=num_hops_dympnn,
                           n_f_atom=n_f_atom, n_f_bond=n_f_bond)
        self.readout = ReadoutNetwork(model_dim=model_dim + n_f_atom, hidden_dim=readout_hidden_dim,
                                      num_heads=readout_num_heads, dropout=dropout)
        self.output_net = OutputNetwork(input_dim=readout_num_heads, hidden_dim=head_hidden_dim,
                                        output_dim=output_dim, num_layers=head_num_layers, dropout=dropout)

    def forward(self, batch):
        batch = self.enc(batch)
        batch = self.readout(batch)
        batch = self.output_net(batch)
        return batch


class OutputNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super(OutputNetwork, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.PReLU(), nn.Dropout(p=dropout)] + \
                 [v for t in list(zip([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)],
                                      [nn.PReLU() for _ in range(num_layers - 2)],
                                      [nn.Dropout(dropout) for _ in range(num_layers - 2)])) for v in t] + \
                 [nn.Linear(hidden_dim, output_dim)]
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


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


class dyMPNN(nn.Module):
    def __init__(self, model_dim: int, num_layers: int, n_f_bond: int, dropout: float):
        super(dyMPNN, self).__init__()

        self.layers = [NNConv(in_channels=model_dim,
                              out_channels=model_dim,
                              nn=nn.Linear(in_features=n_f_bond, out_features=model_dim * model_dim),
                              aggr='mean') for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.prelu = nn.PReLU()

    def forward(self, data, num_hops):
        for i, layer in enumerate(self.layers):
            for _ in range(num_hops):
                data.x = layer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
                data.x = self.prelu(data.x)
                data.x = self.dropouts[i](data.x)
        return data


class Aggregate2Node(MessagePassing):
    def __init__(self):
        super(Aggregate2Node, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Start propagating messages.
        return self.propagate(edge_index, x=x)


class Aggregate2Batch(MessagePassing):
    def __init__(self):
        super(Aggregate2Batch, self).__init__(aggr='add')  # "Add" aggregation.
        self.m = None

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 4: Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j has shape [E, out_channels]
        m = torch.cat((x_j, edge_attr), axis=1)
        self.m = m
        return m


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class Encoder(nn.Module):
    def __init__(self, model_dim: int, num_layers: int, num_heads: int, dropout: float,
                 num_layers_dympnn: int, num_hops_dympnn: int,
                 n_f_atom: int, n_f_bond: int):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.num_hops_dympnn = num_hops_dympnn

        self.embedding = nn.Linear(n_f_atom, model_dim)

        dympnn_layers = [dyMPNN(model_dim, num_layers_dympnn, n_f_bond, dropout) for _ in range(num_layers)]
        self.dympnn_layers = nn.ModuleList(dympnn_layers)

        #         sa_layers = [nn.MultiheadAttention(model_dim, num_heads) for _ in range(num_layers)]
        sa_layers = [MultiHeadAttention(model_dim, num_heads, dropout) for _ in range(num_layers)]
        self.sa_layers = nn.ModuleList(sa_layers)

        ln_layers = [LayerNorm(model_dim) for _ in range(num_layers)]
        self.ln_layers = nn.ModuleList(ln_layers)

        self.a2n = Aggregate2Node()
        self.a2n_linear = nn.Linear(model_dim + n_f_atom, model_dim + n_f_atom)
        self.a2n_ln = LayerNorm(model_dim + n_f_atom)

        self.a2b = Aggregate2Batch()
        self.a2b_linear = nn.Linear(model_dim + n_f_bond, model_dim + n_f_bond)
        self.a2b_ln = LayerNorm(model_dim + n_f_bond)

    def forward(self, input_batch):
        batch = input_batch.clone()
        batch.x = self.embedding(batch.x)

        for i in range(self.num_layers):
            dympnn = self.dympnn_layers[i]
            sa = self.sa_layers[i]
            ln = self.ln_layers[i]

            batch = dympnn(batch, num_hops=self.num_hops_dympnn)

            batch = Batch.to_data_list(batch)
            batch_tensor = [data.x for data in batch]
            batch_shapes = [x.shape[0] for x in batch_tensor]
            batch_mask = [torch.ones_like(x) for x in batch_tensor]
            batch_tensor = nn.utils.rnn.pad_sequence(batch_tensor, batch_first=True)
            batch_mask = nn.utils.rnn.pad_sequence(batch_mask, batch_first=True)

            batch_tensor = sa(batch_tensor, batch_tensor, batch_tensor)[0]
            batch_tensor = ln(batch_tensor)
            for i in range(len(batch)):
                batch[i].x = batch_tensor[i, :batch_shapes[i]]
            batch = Batch.from_data_list(batch)

        input_batch.x = torch.cat((batch.x, input_batch.x), axis=1)
        batch_a2n = input_batch
        #         batch_a2b = batch

        batch_a2n_x = self.a2n(batch_a2n.x, batch_a2n.edge_index)

        #         m_node = self.a2b(batch_a2b.x, batch_a2b.edge_index, batch_a2b.edge_attr)
        #         m_edge = self.a2b.m
        #         batch_a2b_edge_attr = m_node[batch.edge_index[1]] - m_edge

        batch_a2n_x = self.a2n_ln(batch_a2n_x + self.a2n_linear(batch_a2n_x))
        #         batch_a2b_edge_attr = self.a2b_ln(batch_a2b_edge_attr + self.a2b_linear(batch_a2b_edge_attr))

        batch_a2n.x = batch_a2n_x
        #         batch_a2b.edge_attr = batch_a2b_edge_attr

        return batch_a2n  # , batch_a2b


class ReadoutNetwork(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super(ReadoutNetwork, self).__init__()

        self.W1 = nn.Linear(model_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, num_heads)

        self.softmax = nn.Softmax(dim=1)

        self.weights_dropout = nn.Dropout(dropout)
        self.tensor_dropout = nn.Dropout(dropout)

    def forward(self, batch):
        input_batch = batch.clone()

        batch.x = self.W1(batch.x)
        batch.x = torch.tanh(batch.x)
        batch.x = self.W2(batch.x)

        batch = Batch.to_data_list(batch)
        input_batch = Batch.to_data_list(input_batch)

        batch_tensor = [data.x for data in input_batch]
        batch_weights = [data.x for data in batch]
        batch_mask = [torch.ones_like(x) for x in batch_weights]

        batch_tensor = nn.utils.rnn.pad_sequence(batch_tensor, batch_first=True)
        batch_weights = nn.utils.rnn.pad_sequence(batch_weights, batch_first=True)
        batch_mask = nn.utils.rnn.pad_sequence(batch_mask, batch_first=True)

        batch_weights = self.weights_dropout(batch_weights)
        batch_tensor = self.tensor_dropout(batch_tensor)

        batch_weights = batch_weights.masked_fill(batch_mask == 0, -1e9)
        batch_weights = self.softmax(batch_weights)

        batch_tensor = torch.matmul(batch_weights.transpose(1, 2), batch_tensor).sum(axis=2)

        return batch_tensor
