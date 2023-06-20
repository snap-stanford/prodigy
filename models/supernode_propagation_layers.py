import torch
import torch_geometric as pyg
from torch_scatter import scatter, scatter_sum
from models.layer_classes import SupernodeToBgGraphLayer, SupernodeAggrLayer
from models.metaGNN import MetaGNNNoEdgeAttr
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

class AggregateTwoEmbeddings(torch.nn.Module):
    '''
    A simple MLP that takes in two embeddings and outputs a single one. Hopefully will be more expressive than simple
    addition.
    '''
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, emb1, emb2):
        return self.mlp(torch.cat((emb1, emb2), 1))


class SupernodeToBgGraphPropagator(torch.nn.Module, SupernodeToBgGraphLayer):
    '''
    A very simple module that propagates supernode embeddings (obtained from the metagraph) back
    to the background graphs.
    New supernode attributes are projected and added to the background graph node attributes.
    '''

    def __init__(self, emb_dim):
        super(SupernodeToBgGraphPropagator, self).__init__()
        self.proj_sn_attr = torch.nn.Linear(emb_dim, emb_dim)
        self.proj_sn_attr_2 = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x, new_supernode_x, supernode_edge_index, supernode_idx, graph_batch):
        '''
            :param x: node embeddings of the background graph
            :param new_supernode_x: node embeddings of the metagraph
            :param supernode_edge_index: edge index of the metagraph
            :param supernode_idx: the indices of the supernodes in the background graph
        '''
        x[supernode_idx] += self.proj_sn_attr(new_supernode_x)
        # Propagate knowledge back to the original node embeddings
        x[supernode_edge_index[0]] += self.proj_sn_attr_2(x[supernode_edge_index[1]])
        return x


class SupernodeToBgGraphGlobalPropagator(torch.nn.Module, SupernodeToBgGraphLayer):
    '''
    A module that propagates supernode embeddings (obtained from the metagraph) back
    to the background graphs - using a GAT back to both the nodes connected to supernode, as well as all the other nodes
    in subgraphs.
    '''

    def __init__(self, emb_dim):
        super(SupernodeToBgGraphGlobalPropagator, self).__init__()
        self.aggr_sn = AggregateTwoEmbeddings(emb_dim)
        self.aggr_sn_2 = AggregateTwoEmbeddings(emb_dim)
        self.down_gat = MetaGNNNoEdgeAttr(emb_dim=emb_dim, heads=2, n_layers=1)

    def forward(self, x, new_supernode_x, supernode_edge_index, supernode_idx, graph_batch):
        '''
            :param x: node embeddings of the background graph
            :param new_supernode_x: node embeddings of the metagraph
            :param supernode_edge_index: edge index of the metagraph
            :param supernode_idx: the indices of the supernodes in the background graph
        '''
        graph_batch = graph_batch.to(x.device)
        global_edge_index = torch.stack([supernode_idx[graph_batch], torch.arange(x.shape[0]).to(x.device)])
        # edge index supernode -> all nodes in the subgraphs
        x[supernode_idx] = self.aggr_sn(x[supernode_idx], new_supernode_x)
        # Propagate knowledge back to the original node embeddings
        x[supernode_edge_index[0]] = self.aggr_sn_2(x[supernode_edge_index[0]], x[supernode_edge_index[1]])
        # Propagate via "global" edges
        x = self.down_gat(x, global_edge_index)
        return x  # This layer works two ways! (both up and down in a way)


class BgGraphToSupernodePropagator(torch.nn.Module, SupernodeAggrLayer):
    def __init__(self, aggr='mean'):
        super().__init__()
        self.aggr = aggr

    def forward(self, all_node_emb, supernode_edge_index, supernode_idx, graph_batch):
        '''
        A simple aggregator to obtain supernode embeddings.
        :param all_node_emb: X matrix of all nodes in the graph
        :param supernode_edge_index: Edge index of edges from the background (sub)graphs to supernodes
        :param supernode_idx: Idx of the supernodes in the graph
        :return:
        '''
        return scatter(src=all_node_emb[supernode_edge_index[0]], index=supernode_edge_index[1], dim=0, reduce=self.aggr)[
               supernode_idx, :]


class BgGraphToSupernodePropagatorPool(torch.nn.Module, SupernodeAggrLayer):
    def __init__(self, emb_dim, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        self.proj = torch.nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, all_node_emb, supernode_edge_index, supernode_idx, graph_batch):
        '''
        A simple aggregator to obtain supernode embeddings.
        :param all_node_emb: X matrix of all nodes in the graph
        :param supernode_edge_index: Edge index of edges from the background (sub)graphs to supernodes
        :param supernode_idx: Idx of the supernodes in the graph
        :return:
        '''
        graph_emb = scatter(src=all_node_emb[supernode_edge_index[0]], index=supernode_edge_index[1], dim=0, reduce=self.aggr)[
               supernode_idx, :]
        pool = global_max_pool(all_node_emb, graph_batch)
        graph_emb = torch.cat([graph_emb, pool], 1)
        return self.proj(graph_emb)


class BgGraphToSupernodePropagatorCat(torch.nn.Module, SupernodeAggrLayer):
    def __init__(self, emb_dim, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        self.proj = torch.nn.Linear(3 * emb_dim, emb_dim)

    def forward(self, all_node_emb, supernode_edge_index, supernode_idx, graph_batch):
        '''
        A simple aggregator to obtain supernode embeddings.
        :param all_node_emb: X matrix of all nodes in the graph
        :param supernode_edge_index: Edge index of edges from the background (sub)graphs to supernodes
        :param supernode_idx: Idx of the supernodes in the graph
        :return:
        '''
        # assumed for KG!!!

        batch_num_nodes = scatter_sum(torch.ones(graph_batch.shape).to(graph_batch.device), graph_batch)
        head_idxs = torch.cumsum(torch.cat([torch.tensor([0]).to(graph_batch.device),batch_num_nodes[:-1]]), 0).long()
        tail_idxs = torch.cumsum(torch.cat([torch.tensor([0]).to(graph_batch.device),batch_num_nodes[:-1]]), 0).long() + 1
        graph_emb = torch.cat([all_node_emb[head_idxs] , all_node_emb[tail_idxs], global_max_pool(all_node_emb, graph_batch)], 1)
        return self.proj(graph_emb)


class BgGraphToSupernodeGlobalPropagator(torch.nn.Module, SupernodeAggrLayer):
    def __init__(self, emb_dim, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        self.emb_dim = emb_dim
        self.sn_aggr = AggregateTwoEmbeddings(emb_dim)
        # self.up_gat = MetaGNN(edge_attr_dim=0, emb_dim=emb_dim, heads=2, n_layers=1)


    def forward(self, all_node_emb, supernode_edge_index, supernode_idx, graph_batch):
        '''
        A simple aggregator to obtain supernode embeddings.
        :param all_node_emb: X matrix of all nodes in the graph
        :param supernode_edge_index: Edge index of edges from the background (sub)graphs to supernodes
        :param supernode_idx: Idx of the supernodes in the graph
        :return:
        '''
        ## TODO.

