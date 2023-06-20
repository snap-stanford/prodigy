from typing import Optional, List, NamedTuple, Tuple, Union
import torch
import os
from tqdm import trange

from torch import Tensor
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor, coalesce


def preprocess(edge_index, num_nodes=None, bidirectional=True):
    N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
    edge_attr = torch.arange(edge_index.size(1))
    if bidirectional:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, -1 - edge_attr], dim=0)
    whole_adj = SparseTensor.from_edge_index(edge_index, edge_attr, (N, N), is_sorted=False)

    rowptr, col, value = whole_adj.csr()  # convert to csr form
    whole_adj = SparseTensor(rowptr=rowptr, col=col, value=value, sparse_sizes=(N, N), is_sorted=True, trust_data=True)
    return whole_adj


def sample_k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    whole_adj: Tensor,
    num_nodes: Optional[int] = None,
    bidirectional: bool = True,
    size: int = 100,
    limit: int = 2000,
):
    '''
    input: similar to k_hop_subgraph (check https://pytorch-geometric.readthedocs.io/en/1.5.0/modules/utils.html?highlight=subgraph#torch_geometric.utils.k_hop_subgraph)
      key difference:
        (1) we need preprocess function and achieve the adj (of type SparseTensor)
        (2) argument `size` is added to control the number of nodes in the sampled subgraph
    output:
      n_id: the nodes involved in the subgraph
      edge_index: the edge_index in the subgraph
      edge_id: the id of edges in the subgraph
    '''
    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx])
    elif isinstance(node_idx, list):
        node_idx = torch.tensor(node_idx)

    assert isinstance(whole_adj, SparseTensor)

    adjs = []
    for _ in range(num_hops):
        adj, node_idx = whole_adj.sample_adj(node_idx, size, replace=False)
        adjs.append(adj.coo())
        if node_idx.size(0) >= limit:
            break
    row = torch.cat([adj[0] for adj in adjs])
    col = torch.cat([adj[1] for adj in adjs])
    e_id = torch.cat([adj[2] for adj in adjs])

    if node_idx.size(0) > limit:
        node_idx = node_idx[:limit]
        mask = (row < limit).logical_and(col < limit)
        row = row[mask]
        col = col[mask]
        e_id = e_id[mask]

    mask = e_id < 0
    row[mask], col[mask] = col[mask], row[mask]
    e_id[mask] = -e_id[mask] - 1
    edge_index = torch.stack([row, col], dim=0)

    node_count = node_idx.size(0)
    edge_index, e_id = coalesce(edge_index, e_id, node_count, node_count, "min")

    return node_idx, edge_index, e_id


class NeighborSampler:
    def __init__(
        self,
        graph, # pyg graph
        num_hops: int,
        size: int = 100,
        limit: int = 2000,
    ):
        self.num_hops = num_hops
        self.size = size
        self.limit = limit
        self.whole_adj = preprocess(graph.edge_index, graph.num_nodes)
        self.whole_adj.share_memory_()

    def sample_node(self, node_idx):
        return sample_k_hop_subgraph(
            node_idx,
            num_hops=self.num_hops,
            whole_adj=self.whole_adj,
            size=self.size,
            limit=self.limit,
        )
    
    def sample_edge(
        self,
        node_idx: Tensor,
        direction: str,
    ):
        """
        direction: "in", "out", "inout"
        return sampled edges that contain node_idx
        order of edges will change!!
        """
        node_idx, edge_index, e_id =  sample_k_hop_subgraph(
            node_idx,
            num_hops=1,
            whole_adj=self.whole_adj,
            size=1,
            limit=3*len(node_idx),
        )
        rev = edge_index[0] > edge_index[1]
        e_id[rev] = -e_id[rev] - 1
        return e_id

    def random_walk(
        self,
        node_idx: Tensor,
        direction: str,
    ):
        """
        direction: "in", "out", "inout"
        """
        rowptr, col, e_id = self.whole_adj.csr()
        for _ in range(self.num_hops):
            row_start = rowptr[node_idx]
            row_end = rowptr[node_idx + 1]
            idx = (torch.rand(node_idx.shape) * (row_end - row_start)).long() + row_start
            node_idx = col[idx]
            mask = row_start < row_end
            if direction == "in":
                mask = mask.logical_and(e_id[idx] < 0)
            elif direction == "out":
                mask = mask.logical_and(e_id[idx] >= 0)
            node_idx = node_idx[mask]
        return node_idx


class NeighborSamplerCacheAdj(NeighborSampler):
    def __init__(
        self,
        cache_path,
        graph, # pyg graph
        num_hops: int,
        size: int = 100,
        limit: int = 2000,
    ):
        self.num_hops = num_hops
        self.size = size
        self.limit = limit
        if os.path.exists(cache_path):
            print(f"Loading adjacent matrix for neighbor sampling from {cache_path}")
            self.whole_adj = torch.load(cache_path)
            print(f"Loaded adjacent matrix for neighbor sampling from {cache_path}")
        else:
            print(f"Preprocessing adjacent matrix for neighbor sampling")
            self.whole_adj = preprocess(graph.edge_index, graph.num_nodes)
            print(f"Saving adjacent matrix for neighbor sampling to {cache_path}")
            torch.save(self.whole_adj, cache_path)
            print(f"Saved adjacent matrix for neighbor sampling to {cache_path}")
        self.whole_adj.share_memory_()


if __name__ == '__main__':
    from ogb.lsc import MAG240MDataset, MAG240MEvaluator
    ROOT = '<DATA ROOT>'
    dataset = MAG240MDataset(ROOT)

    import numpy as np
    import torch
    from torch_geometric.data import Data

    edge_index = dataset.edge_index('paper', 'cites', 'paper')
    edge_index = torch.from_numpy(edge_index)
    print("Dataset loaded")
    graph = Data(edge_index=edge_index)
    print(graph)
    print(graph.num_nodes)

    nh = 2
    sampler = NeighborSamplerCacheAdj("test_adj.pt", graph, nh, relabel_nodes=True)

    num_nodes = graph.num_nodes # this is an expensive operation
    for i in trange(10000):
        a = np.random.choice(num_nodes)
        sampler.sample_node(i)
