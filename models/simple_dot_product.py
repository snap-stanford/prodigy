# Simple dot product decoder - used for debugging the initial start acc

import torch
import torch_geometric as pyg
import numpy as np
from models.layer_classes import MetagraphLayer, SupernodeAggrLayer, SupernodeToBgGraphLayer, BackgroundGNNLayer
from torch_scatter import scatter_mean

class SimpleDotProdModel(torch.nn.Module):
    def __init__(self, layer_list, initial_label_mlp=torch.nn.Identity(), initial_input_mlp=torch.nn.Identity(),
                 final_label_mlp=torch.nn.Identity(), final_input_mlp=torch.nn.Identity(),
                 params=None, text_dropout=None):
        super().__init__()
        self.random_mlp = torch.nn.Linear(768, 768)
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if params is not None:
            self.params = params

    def decode(self, input_x, label_x, metagraph_edge_index, edgelist_bipartite=False):
        if edgelist_bipartite:
            ind0 = metagraph_edge_index[0, :]
            ind1 = metagraph_edge_index[1, :]
            decoded_logits = (self.cos(input_x[ind0], label_x[ind1]) + 1) / 2
            return decoded_logits
        # strip last 2 features from input_x if necessary
        if input_x.shape[1] > label_x.shape[1]:
            input_x = input_x[:, :-2]
            print("temporarily stripping off the last 2 features from input_x")
        x = torch.cat((input_x, label_x))
        ind0 = metagraph_edge_index[0, :]
        ind1 = metagraph_edge_index[1, :]
        decoded_logits = self.cos(x[ind0], x[ind1]) * self.logit_scale.exp()
        return decoded_logits

    def forward(self, graph, x_label, y_true_matrix, metagraph_edge_index, metagraph_edge_attr, query_set_mask, input_seqs=None, query_seqs=None, query_seqs_gt=None, task_mask=None):
        '''
        Params as returned by the batching function.
        # task_mask: Not actually needed here, but is passed here from the dataloader batch output..
        :return: y_true_matrix, y_pred_matrix (for the query set only!)
        '''
        supernode_idx = graph.supernode + graph.ptr[:-1]
        supernode_edge_index = graph.edge_index_supernode
        x_input = scatter_mean(src=graph.x[supernode_edge_index[0]], index=supernode_edge_index[1], dim=0)[supernode_idx, :]
        y_pred_matrix = self.decode(x_input, x_label, metagraph_edge_index, edgelist_bipartite=False).reshape(
            y_true_matrix.shape)

        qry_idx = torch.where(query_set_mask.reshape(-1, y_true_matrix.shape[1])[:, 0] == 1)[0]
        return y_true_matrix[qry_idx, :], y_pred_matrix[qry_idx, :], graph

