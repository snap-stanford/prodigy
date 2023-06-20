import torch
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops
from torch.nn import functional as F
import math
import torch.nn.functional as F
from torch import nn
from models.layer_classes import MetagraphLayer
from torch_scatter import scatter_mean
from torch.nn import TransformerEncoder, TransformerEncoderLayer




def custom_attn(self, query, key, value, attention_mask=None, head_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.tensor(
            value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)

        #############customized part ###############
        # attention_mask represents the queries
        # so nothing attend to queries

        # make query attend to itself by setting
        attention_mask = attention_mask.repeat(1,1,attention_mask.shape[-1], 1)
        attention_mask = attention_mask * (1 - torch.eye(attention_mask.shape[-1]).to(attention_mask.device) )[None, None]


        # support can attend to each other
        m = attention_mask + attention_mask.transpose(3, 2)
        causal_mask = torch.where(m.to(causal_mask.device) == 0, True, causal_mask)
        ##############################################

        # attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights

# GPT2Attention._attn = custom_attn



class MetaGNNLayer(MessagePassing):
    """
    GAT gnn for bipartite graph.
    Args:
        edge_attr_dim (int): dimension of edge attributes (2 for metagraph)
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        aggr (str): aggregation method. Default: "add".
    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, edge_attr_dim, emb_dim, heads=1, dropout=0, aggr="add", batch_norm=True):
        super(MetaGNNLayer, self).__init__()
        # k, q, v matrices, no bias for now
        self.heads = heads
        self.head_dim = emb_dim // heads

        self.mlp_kqv = torch.nn.Linear(emb_dim, 3 * emb_dim)

        self.emb_dim = emb_dim
        self.lin_edge = torch.nn.Linear(edge_attr_dim, emb_dim)
        self.att_mlp = torch.nn.Sequential(torch.nn.Linear(3 * self.head_dim, self.head_dim), torch.nn.ReLU(), torch.nn.Linear(self.head_dim, 1))
        self.out_proj = torch.nn.Linear(emb_dim, emb_dim)

        self.dropout = dropout
        self.aggr = aggr
        self.bn = torch.nn.Identity()
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(emb_dim)

    def forward(self, x, edge_index, edge_attr=None, start_right=None):
        kqv_x = self.mlp_kqv(x)
        out = self.propagate(edge_index, x=kqv_x, edge_attr=edge_attr, size=None)

        out = F.dropout(out, p=self.dropout, training=self.training) + x
        out = self.bn(out)
        return out

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(F.relu(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def message(self, x_j, x_i , edge_attr, index, ptr, size_i):

        H, E = self.heads, self.head_dim

        # compute query of target; k,v of source
        q = x_i[:, :self.emb_dim].reshape(-1, H, E)
        k = x_j[:, self.emb_dim: 2 * self.emb_dim].reshape(-1, H, E) / math.sqrt(E)
        v = x_j[:, 2 * self.emb_dim: 3*self.emb_dim].reshape(-1, H, E)

        # apply linear layer to edge
        edge_attr = self.lin_edge(edge_attr)
        edge_attr = edge_attr.view(edge_attr.shape[0], H, E)

        # apply mlp to compute attention score
        alpha = self.att_mlp(torch.cat([k, q, F.relu(edge_attr)], dim = -1))
        # alpha = (k*q).sum(-1).unsqueeze(2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        attn_output = alpha * v

        attn_output = attn_output.view(attn_output.shape[0], H * E)
        attn_output = self.out_proj(attn_output)

        return attn_output

class MetaGATConvLayer(MessagePassing):
    def __init__(self, edge_attr_dim, emb_dim, heads=8, dropout=0, self_loops=True, norm=True):
        super(MetaGATConvLayer, self).__init__()
        
        self.gat = GATv2Conv(in_channels=emb_dim, out_channels=emb_dim, heads=heads, edge_dim=edge_attr_dim, add_self_loops=self_loops)

        self.head_proj = torch.nn.Linear(emb_dim * heads, emb_dim)
        # self.edge_proj = torch.nn.Linear(edge_attr_dim, emb_dim) # can also try with mlp
        # self.label_proj = torch.nn.Linear(emb_dim, emb_dim)

        self.mlpf = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * 2), torch.nn.GELU(), torch.nn.Linear(emb_dim * 2, emb_dim))
        
        self.ln1 = torch.nn.Identity()
        self.ln2 = torch.nn.Identity()
        if norm:
            self.ln1 = torch.nn.LayerNorm(emb_dim)
            self.ln2 = torch.nn.LayerNorm(emb_dim)
    
    def forward(self, x, edge_index, edge_attr=None, start_right=None):
        # Edge projection (commented out bc it doesnt improve)
        # if edge_attr is not None:
        #     edge_attr = self.edge_proj(edge_attr)

        # Label projection (commented out bc it doesnt improve)
        # Need to figure out which part of x is subgraph and which part is label
        # project X[start_right:] to emb_dim
        # label_emb = self.label_proj(x[start_right:])
        # x = torch.cat([x[:start_right], label_emb], dim=0)
        
        x = x + self.head_proj(self.gat(self.ln1(x), edge_index, edge_attr=edge_attr))
        x = x + self.mlpf(self.ln2(x))
        return x

class MetaGATConvLayerBi(MessagePassing):
    def __init__(self, edge_attr_dim, emb_dim, heads=8, dropout=0, self_loops=True, norm=True):
        super(MetaGATConvLayerBi, self).__init__()
        
        self.gat_node = GATv2Conv(in_channels=emb_dim, out_channels=emb_dim, heads=heads, edge_dim=edge_attr_dim, add_self_loops=self_loops)
        self.gat_label = GATv2Conv(in_channels=emb_dim, out_channels=emb_dim, heads=heads, edge_dim=edge_attr_dim, add_self_loops=self_loops)

        self.head_proj_node = torch.nn.Linear(emb_dim * heads, emb_dim)
        self.head_proj_label = torch.nn.Linear(emb_dim * heads, emb_dim)

        self.mlpf = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * 2), torch.nn.GELU(), torch.nn.Linear(emb_dim * 2, emb_dim))

        self.ln1 = torch.nn.Identity()
        self.ln2 = torch.nn.Identity()
        if norm:
            self.ln1 = torch.nn.LayerNorm(emb_dim)
            self.ln2 = torch.nn.LayerNorm(emb_dim)
    
    def forward(self, x, edge_index, edge_attr=None, start_right=None):

        x = self.ln1(x)
        x_node = self.head_proj_node(self.gat_node(x, edge_index, edge_attr=edge_attr)[:start_right])
        x_label = self.head_proj_label(self.gat_label(x, edge_index, edge_attr=edge_attr)[start_right:])

        x = x + torch.cat([x_node, x_label], dim=0)
        x = x + self.mlpf(self.ln2(x))

        return x
        

class MetaAverage(torch.nn.Module, MetagraphLayer):
    def __init__(self, edge_attr_dim, emb_dim, heads = 2, n_layers=1, dropout = 0, aggr="add"):
        super().__init__()
        
    def forward(self, x, edge_index, edge_attr, query_mask, start_right, input_seqs, query_seqs, query_seqs_gt, prev_hidden_states = None, **kwargs):

        inputs_ids = input_seqs
        inputs_embeds = x[inputs_ids[:, ::2]]

        col_class = inputs_ids[:, 1::2].reshape(-1)
        averaged_support = scatter_mean(inputs_embeds.reshape(col_class.shape[0], -1)[col_class < x.shape[0]], col_class[col_class < x.shape[0]], dim=0, dim_size=x.shape[0])

        x = torch.cat([x[:start_right], averaged_support[start_right:]], dim = 0)
        return x
 

class MetaGNN(torch.nn.Module, MetagraphLayer):
    def __init__(self, edge_attr_dim, emb_dim, heads = 8, n_layers=1, dropout = 0, aggr="add", has_final_back=False, msg_pos_only = False, self_loops = True, batch_norm = True, gat_layer = False, use_relu=False):
        super().__init__()
        self.num_gnn_layers = n_layers
        self.gnn_layers = torch.nn.ModuleList()
        self.msg_pos_only = msg_pos_only
        self.self_loops = self_loops
        
        if gat_layer:
            self.add_layers_gat(emb_dim, heads, edge_attr_dim, dropout, batch_norm, has_final_back)
        else:
            self.add_layers_original(emb_dim, heads, edge_attr_dim, dropout, batch_norm, has_final_back)
        if use_relu:
            self.gnn_non_linear = torch.nn.ReLU()
        else:
            self.gnn_non_linear = torch.nn.GELU()

    def add_layers_original(self, emb_dim, heads, edge_attr_dim, dropout, batch_norm, has_final_back):
        self.gnn_layers_back = MetaGNNLayer(emb_dim=emb_dim, heads=heads, edge_attr_dim=edge_attr_dim, batch_norm=batch_norm) if has_final_back else None
        for i in range(self.num_gnn_layers):
            self.gnn_layers.append(MetaGNNLayer(emb_dim=emb_dim, heads=heads, edge_attr_dim=edge_attr_dim, dropout=dropout, batch_norm=batch_norm))
    
    def add_layers_gat(self, emb_dim, heads, edge_attr_dim, dropout, batch_norm, has_final_back):
        self.gnn_layers_back = MetaGATConvLayerBi(edge_attr_dim, emb_dim, heads=heads, dropout=dropout, self_loops=self.self_loops, norm=batch_norm) if has_final_back else None

        for i in range(self.num_gnn_layers):
            self.gnn_layers.append(MetaGATConvLayerBi(edge_attr_dim, emb_dim, heads=heads, dropout=dropout, self_loops=self.self_loops, norm=batch_norm))

    def forward(self, x, edge_index, edge_attr, query_mask, start_right, **kwargs):
        '''
        :param x: Feature matrix
        :param edge_index: Edge index for the bipartite graph.
        :param edge_attr: Edge attributes for the bipartite graph.
        :return:
        '''
        if not query_mask.type() == "torch.BoolTensor":
            query_mask = query_mask.bool()
        if not self.msg_pos_only:
            support_mask = ~query_mask
        else:
            positives = edge_attr[:, -1] == 1  # remove negatives
            support_mask = (~query_mask) & positives

        query_in_mask = query_mask
        edge_index_back = edge_index[:, query_in_mask].flip(0)
        edge_attr_back = edge_attr[query_in_mask]

        edge_index = torch.cat([edge_index[:, support_mask], edge_index.flip(0)], 1)
        edge_attr =  torch.cat([edge_attr[support_mask], edge_attr], 0)
        
        # add_self_loops:
        if self.self_loops:
            num_nodes = x.size(0)
            edge_index, edge_attr = remove_self_loops(
               edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
               edge_index, edge_attr, fill_value=torch.tensor([0, 0]).to(edge_attr.device),
               num_nodes=num_nodes)
        
        for i in range(self.num_gnn_layers):
            # hack for heterogeneous graph; should be fixed
            x = self.gnn_layers[i](x, edge_index, edge_attr=edge_attr, start_right=start_right)
            if i != self.num_gnn_layers - 1:
                x = self.gnn_non_linear(x)
        if self.gnn_layers_back is not None:
            x = self.gnn_non_linear(x)
            x = self.gnn_layers_back(x, edge_index_back, edge_attr=edge_attr_back, start_right=start_right)
        return x



class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.n_embd
        dropout = 0
        nhead = config.n_head

        self.pos_encoder = PositionalEncoding(64, dropout)
        encoder_layers = TransformerEncoderLayer(d_model + 64 + 256, nhead, d_model, dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.n_layer)

        self.d_model = d_model


    def forward(self, src, src_mask, position_ids = None, pe_type = "sin"):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        src = self.pos_encoder(src, position_ids, pe_type)

        output = src
        for mod in self.transformer_encoder.layers:
            residual = output
            output = mod(output, src_mask=src_mask)
            output += residual

        output = output[:,:,:self.d_model]

        return output



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.wpe = nn.Embedding(max_len, d_model)
        self.in_out = nn.Embedding(2, 256)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None, pe_type = "sin"):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # add random offset
#         offset = np.random.choice((200 - x.size(1) + 1))
        offset = 0
        if pe_type == "sin":
            if position_ids is None:
                x = torch.cat([x, self.pe[:, :x.size(1)].repeat(x.shape[0], 1, 1)], -1)
            else:
                x = torch.cat([x, self.pe[:, position_ids].repeat(x.shape[0], 1, 1)], -1)
            x = torch.cat([x, self.in_out(torch.tensor([0,1]).repeat(x.size(1)//2).to(x.device)).unsqueeze(0).repeat(x.shape[0], 1, 1)], -1)
        else:
            x = torch.cat([x, self.wpe(position_ids).unsqueeze(0).repeat(x.shape[0], 1, 1), self.in_out(torch.tensor([0,1]).repeat(x.size(1)//2).to(x.device)).unsqueeze(0).repeat(x.shape[0], 1, 1)], -1)
        # x = x + self.pe[:, :x.size(1)]
        return x



def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class MetaTransformerPytorch(torch.nn.Module, MetagraphLayer):
    def __init__(self, config, attention_mask_scheme):
        super().__init__()
        self.transformer = TransformerModel(config)
        self.masked_token_emb = torch.zeros([1, config.n_embd])
        self.false_token_emb = torch.zeros([1, config.n_embd])
        self.attention_mask_scheme = attention_mask_scheme


    def forward(self, x, edge_index, edge_attr, query_mask, start_right, input_seqs, query_seqs, query_seqs_gt, prev_hidden_states = None, **kwargs):
        '''
        :param x: Feature matrix
        :param edge_index: Edge index for the bipartite graph.
        :param edge_attr: Edge attributes for the bipartite graph.
        :return:
        '''
        #TODO: label can come from prev_hidden_states if prev_hidden_states is not None
        x = torch.cat([x, self.masked_token_emb.to(x.device), self.false_token_emb.to(x.device) ], 0)
        inputs_ids = torch.cat([input_seqs, query_seqs], dim = 1)
        inputs_embeds = x[inputs_ids]

        seq_len = inputs_ids.shape[1]

        src_mask = None
        position_ids = None
        pe_type = "sin"

        if self.attention_mask_scheme == "none":
            src_mask = None
        elif self.attention_mask_scheme == "causal":
            src_mask = generate_square_subsequent_mask(seq_len).to(inputs_ids.device)
        elif self.attention_mask_scheme == "mask":
            src_mask = generate_square_subsequent_mask(seq_len).to(inputs_ids.device)

            # mask query tokens for all tokens
            attention_mask = torch.zeros((seq_len, seq_len)).to(inputs_ids.device)
            attention_mask[:, input_seqs.shape[1]+1::2] = 1
            src_mask = torch.where(attention_mask == 1,float('-inf'), 0) + src_mask

        elif self.attention_mask_scheme == "special":
            src_mask = generate_square_subsequent_mask(seq_len).to(inputs_ids.device)

            # mask query tokens for all tokens
            attention_mask = torch.zeros((seq_len, seq_len)).to(inputs_ids.device)
            attention_mask[:, input_seqs.shape[1]:] = 1

            # make query attend to itself
            attention_mask = attention_mask * (1 - torch.eye(attention_mask.shape[-1]).to(attention_mask.device) )

            src_mask = torch.where(attention_mask == 1,float('-inf'), 0) + src_mask # optimization becomes bad if remove this src_mask

            # position_ids_1 = torch.arange(0, input_seqs.shape[-1], dtype=torch.long, device=x.device)
            # position_ids_2 = torch.tensor([input_seqs.shape[-1], input_seqs.shape[-1]+1], dtype=torch.long, device=x.device).repeat(query_seqs.shape[1]//2)
            # position_ids = torch.cat([position_ids_1, position_ids_2])

            position_ids_1 = torch.arange(0, input_seqs.shape[-1]//2, dtype=torch.long, device=x.device).repeat_interleave(2)
            position_ids_2 = torch.tensor([input_seqs.shape[-1]], dtype=torch.long, device=x.device).repeat(query_seqs.shape[1])
            position_ids = torch.cat([position_ids_1, position_ids_2])
            # pe_type = "wpe"

        hidden_states = self.transformer(
            inputs_embeds,
            src_mask,
            position_ids = position_ids,
            pe_type = pe_type
        )
        # hidden_states = hidden_states.reshape(inputs_embeds.shape[0], inputs_embeds.shape[1], inputs_embeds.shape[2])
        col = inputs_ids.reshape(-1)
        x_label = scatter_mean(hidden_states.reshape(col.shape[0], -1), col, dim=0, dim_size=x.shape[0])[:-2]

        # x_pred_label = scatter_mean(hidden_states.reshape(col.shape[0], -1)[1:], col[:-1], dim=0, dim_size=x.shape[0])[:-1]

        return x_label


class MetaTransformer(torch.nn.Module, MetagraphLayer):
    def __init__(self, Transformer_cls, config):
        super().__init__()
        self.transformer = Transformer_cls(config)
        self.masked_token_emb = torch.zeros([1, config.n_embd])


    def forward(self, x, edge_index, edge_attr, query_mask, start_right, input_seqs, query_seqs, query_seqs_gt, prev_hidden_states = None, **kwargs):
        '''
        :param x: Feature matrix
        :param edge_index: Edge index for the bipartite graph.
        :param edge_attr: Edge attributes for the bipartite graph.
        :return:
        '''

        #TODO: label can come from prev_hidden_states if prev_hidden_states is not None
        x = torch.cat([x, self.masked_token_emb.to(x.device) ], 0)
        inputs_ids = torch.cat([input_seqs, query_seqs], dim = 1)
        inputs_embeds = x[inputs_ids]

        attention_mask = None
        attention_mask = torch.ones(inputs_ids.shape)
        attention_mask[:, input_seqs.shape[1]+1::2] = 0

        position_ids = torch.arange(0, inputs_ids.shape[-1]//2, dtype=torch.long, device=x.device).repeat_interleave(2)
        position_ids = position_ids.unsqueeze(0).view(-1, inputs_ids.shape[-1])

        transformer_outputs = self.transformer(
            attention_mask=attention_mask.to(inputs_embeds.device),
            position_ids = position_ids,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        col = inputs_ids.reshape(-1)
        x = scatter_mean(hidden_states.reshape(col.shape[0], -1), col, dim=0, dim_size=x.shape[0])

        return x[:-1]
        # , hidden_states


class MetaGNNLayerNoEdgeAttr(MessagePassing):
    """
    GAT gnn for bipartite graph.
    Args:
        edge_attr_dim (int): dimension of edge attributes (2 for metagraph)
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        aggr (str): aggregation method. Default: "add".
    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, heads=1, dropout=0, aggr="add"):
        super(MetaGNNLayerNoEdgeAttr, self).__init__()
        # k, q, v matrices, no bias for now
        self.heads = heads
        self.head_dim = emb_dim // heads

        self.mlp_kqv = torch.nn.Linear(emb_dim, 3 * emb_dim)
        self.emb_dim = emb_dim
        self.att_mlp = torch.nn.Sequential(torch.nn.Linear(2 * self.head_dim, self.head_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(self.head_dim, 1))
        self.out_proj = torch.nn.Linear(emb_dim, emb_dim)

        self.dropout = dropout
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr=None, start_right=None):
        # add self loops?

        kqv_x = self.mlp_kqv(x)
        out = self.propagate(edge_index, x=kqv_x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_j, x_i, index, ptr, size_i):
        H, E = self.heads, self.head_dim

        # compute query of target; k,v of source
        q = x_i[:, :self.emb_dim].reshape(-1, H, E)
        k = x_j[:, self.emb_dim: 2 * self.emb_dim].reshape(-1, H, E) / math.sqrt(E)
        v = x_j[:, 2 * self.emb_dim: 3 * self.emb_dim].reshape(-1, H, E)

        # apply linear layer to edge

        # apply mlp to compute attention score
        alpha = self.att_mlp(torch.cat([k, q], dim=-1))
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        attn_output = alpha * v

        attn_output = attn_output.view(attn_output.shape[0], H * E)
        attn_output = self.out_proj(attn_output)

        return attn_output


class MetaGNNNoEdgeAttr(torch.nn.Module, MetagraphLayer):
    def __init__(self, emb_dim, heads=2, n_layers=1):
        super().__init__()
        self.num_gnn_layers = n_layers
        self.gnn_layers = torch.nn.ModuleList()
        # self.gnn_layers_back = torch.nn.ModuleList()
        for i in range(self.num_gnn_layers):
            self.gnn_layers.append(MetaGNNLayerNoEdgeAttr(emb_dim=emb_dim, heads=heads))
            # self.gnn_layers_back.append(MetaGNNLayer(emb_dim=emb_dim, heads=heads, edge_attr_dim=edge_attr_dim))

        self.gnn_non_linear = torch.nn.ReLU()

    def forward(self, x, edge_index):
        '''
        :param x: Feature matrix
        :param edge_index: Edge index for the bipartite graph.
        :param edge_attr: Edge attributes for the bipartite graph.
        :return:
        '''

        edge_index = torch.cat([edge_index, edge_index.flip(0)], 1)
        # edge_index_t = edge_index.flip(0)
        # edge_attr = torch.cat([edge_attr, edge_attr], 0)

        # add_self_loops:
        num_nodes = x.size(0)
        edge_index, edge_attr = remove_self_loops(
            edge_index, edge_attr=None)
        edge_index, _ = add_self_loops(
            edge_index, edge_attr=None,
            num_nodes=num_nodes)

        x_prev = x
        for i in range(self.num_gnn_layers):
            # hack for heterogeneous graph; should be fixed
            x = self.gnn_layers[i](x, edge_index)
            # + self.gnn_layers_back[i](x, edge_index_t, edge_attr=edge_attr)
            x = self.gnn_non_linear(x) + x_prev
            x_prev = x
        return x


