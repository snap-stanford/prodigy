'''
    Classes for creating a multilayer GNN model.
'''

import torch
from models.layer_classes import BackgroundGNNLayer
from torch_geometric.nn import global_mean_pool

class MultiLayerGNN(torch.nn.Module, BackgroundGNNLayer):
    def __init__(self, module_list: torch.nn.ModuleList, supernode_gnn=None, reset_after_layer = None, emb_dim = 256):
        '''

        :param module_list: ModuleList for each layer's GNN.
        :param supernode_gnn: If not None, will be used to perform message passing on the supernode edges.
        '''
        super().__init__()
        self.module_list = module_list
        self.supernode_gnn = supernode_gnn
        self.act = torch.nn.ReLU()
        self.reset_after_layer = reset_after_layer
        self.reset_mlp = torch.nn.Linear(2*emb_dim, emb_dim)
        self.reset_mlp_c = torch.nn.Linear(emb_dim, emb_dim)
        self.reset_mlp_m = torch.nn.Linear(emb_dim, emb_dim)
        

    def forward(self, x_orig, x, edge_index, edge_attr, supernode_edge_index=None, center_node_index = None, batch = None):
        '''
        If supernode_edge_index is not None, it will also pass messages on the supernode edges.
        :param x:
        :param edge_index:
        :param edge_attr:
        :param supernode_edge_index:
        :return:
        '''
        # orig_x = x
        for idx, module in enumerate(self.module_list):
            # need to add edge attr correspondingly when edge_attr is not None

            if self.reset_after_layer is not None and idx in self.reset_after_layer:
                # reset features other than center node
                # condition = self.reset_mlp_c(x[center_node_index]) + self.reset_mlp_m(global_mean_pool(x = x, batch = batch))
                # condition = self.act(condition)[batch]  
                condition = x[center_node_index][batch]           
                x = self.reset_mlp(torch.cat([x_orig, condition], 1))
                x = self.act(x)

            x = module(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.act(x)

            # if orig_x.shape[1] == x.shape[1]:
            #     x = x + orig_x
            # orig_x = x
        if supernode_edge_index is not None and self.supernode_gnn is not None:
            x = self.supernode_gnn(x=x, edge_index=supernode_edge_index)
        x = x.clone()
        x[center_node_index] =  self.act(self.reset_mlp_c(x[center_node_index]) + self.reset_mlp_m(global_mean_pool(x = x, batch = batch)))
        return x


class MultiLayerBipartiteGNN(torch.nn.Module):
    def __init__(self, module_list: torch.nn.ModuleList, transpose_edges_after_each_iter=True):
        '''
        Multilayer bipartite GNN for multiple message passing layers on the metagraph.
        :param module_list:
        :param transpose_edges_after_each_iter:
        '''
        super().__init__()
        self.module_list = module_list
        self.transpose = transpose_edges_after_each_iter
        self.act = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr, start_right):
        '''
        :param x: Feature matrix
        :param edge_index: Edge index for the bipartite graph.
        :param edge_attr: Edge attributes for the bipartite graph.
        :param start_right: At what node index the right side of the metagraph starts.
        :return:
        '''
        curr_transpose = False
        edge_index_t = edge_index[[1, 0], :]
        for module in self.module_list:
            msg_passing_net = edge_index if not curr_transpose else edge_index_t
            x = module(x=x, edge_index=msg_passing_net, edge_attr=edge_attr, start_right=start_right)
            x = self.act(x)
            if self.transpose:
                curr_transpose = not curr_transpose
        return x

