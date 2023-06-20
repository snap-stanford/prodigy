import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data

class SubgraphDataset(Dataset):
    def __init__(self, graph, neighbor_sampler, offset=0, bidirectional=True, node_graph = False):
        self.graph = graph # torch_geometric.data.Data object of the entire graph
        self.neighbor_sampler = neighbor_sampler #  NeighborSampler object
        self.offset = offset
        self.bidirectional = bidirectional

        self.node_attrs = [key for key, value in self.graph if self.graph.is_node_attr(key)]
        self.edge_attrs = [key for key, value in self.graph if self.graph.is_edge_attr(key) and key != "edge_index"]
        assert not self.edge_attrs or not bidirectional

    def get_subgraph(self, node_idx): # refactor as __item__, add supernode in here
        node_list, edge_index, edge_id = self.neighbor_sampler.sample_node(node_idx)
        data = {}
        data['center_node_idx'] = node_idx
        data['edge_index'] = edge_index
        data['num_nodes'] = len(node_list)
        for key in self.node_attrs:
            data[key] = self.graph[key][node_list]
        for key in self.edge_attrs:
            data[key] = self.graph[key][edge_id]
        if self.bidirectional:
            num_edges = edge_index.size(1)
            data['edge_index'] = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            data['edge_attr'] = torch.cat([
                torch.zeros(num_edges, dtype=torch.float),
                torch.ones(num_edges, dtype=torch.float)]).unsqueeze(1)
        graph = Data(**data)
        return graph

    def add_pooling_supernode(self, data):
        for key in self.node_attrs:
            value = data[key]
            data[key] = torch.cat((value, torch.zeros(1, *value.shape[1:], dtype=value.dtype, layout=value.layout, device=value.device)))

        supernode_idx = data.num_nodes
        data.supernode = torch.tensor([supernode_idx])
        data.edge_index_supernode = torch.tensor([[0], [supernode_idx]], dtype=int)
        data.edge_index_from_supernode = torch.tensor([[supernode_idx], [0]], dtype=int)
        data.num_nodes += 1

    def __getitem__(self, index):
        """
        Returns the subgraph at index and adds the supernode
        """
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]
        elif isinstance(index, tuple):
            return tuple(self.__getitem__(i) for i in index)
        elif isinstance(index, dict):
            return {key: self.__getitem__(value) for key, value in index.items()}
        elif not isinstance(index, int):
            return index

        assert index >= 0 and index < len(self)

        graph = self.get_subgraph(index)

        # Add supernode to the graph
        self.add_pooling_supernode(graph)

        return graph

    def __len__(self):
        return self.graph.num_nodes


class KGSubgraphDataset(Dataset):
    def __init__(self, kg_dataset, neighbor_sampler, sampler_type, node_graph):
        self.sampler_type = sampler_type
        self.neighbor_sampler = neighbor_sampler
        self.name = kg_dataset.dataset
        self.kg_dataset = kg_dataset
        self.ssp_graph = kg_dataset.ssp_graph # torch_geometric.data.Data object of the entire graph
        self.pyg_graph = kg_dataset.graph
        self.hop = kg_dataset.hop
        self.kind = kg_dataset.kind
        self.node_graph = node_graph

        if self.name == "WikiKG90M":
            #self.pyg_graph.x = torch.rand((self.pyg_graph.num_nodes, 768))
            #self.pyg_graph.edge_attr_feat = torch.rand((len(self.pyg_graph.edge_attr), 768))
            print("Will add Wiki features on-the-fly later")
            #  pyg_graph should contain x_id (IDs of nodes) and edge_attr (IDs of relations) - like NELL
            self.pyg_graph.x = None
            self.pyg_graph.edge_attr_feat = None
            edge_types = list(range(max(self.pyg_graph.edge_attr) + 1))
            self.label_embeddings = torch.from_numpy(self.kg_dataset.disk_features["rel"][edge_types]).float()
        elif self.kg_dataset.pretrained_embeddings is not None:
            print("use the provided KG embedddings of relations and entities")
            # use the provided KG embedddings of relations and entities
            self.pyg_graph.x = kg_dataset.pretrained_embeddings["node"][range(self.pyg_graph.num_nodes)]
            self.pyg_graph.edge_attr_feat = kg_dataset.pretrained_embeddings["rel"][self.pyg_graph.edge_attr.flatten().long()]
            lbl_emb = kg_dataset.pretrained_embeddings["rel"][list(range(max(self.pyg_graph.edge_attr) + 1))]
            self.label_embeddings = torch.stack(lbl_emb)
        else:
            x_text = [kg_dataset.id2entity.get(i, str(i)) for i in range(self.pyg_graph.num_nodes)]
            edge_attr_text = [kg_dataset.id2relation.get(i.item(), str(i.item())) for i in self.pyg_graph.edge_attr]
            label_text = [kg_dataset.id2relation.get(i, str(i)) for i in range(max(self.pyg_graph.edge_attr) + 1)]
            if kg_dataset.mid2name is not None:
                x_text = [kg_dataset.mid2name.get(i, i) for i in x_text]
                label_text = [kg_dataset.mid2name.get(i, i) for i in label_text]
            if hasattr(kg_dataset, "text_dict"):
                print("Has text_dict - replacing Q and P ids with text")
                edge_attr_text = [kg_dataset.text_dict.get(i, i) for i in edge_attr_text]
                x_text = [kg_dataset.text_dict.get(i, i) for i in x_text]
                label_text = [kg_dataset.text_dict.get(i, i) for i in label_text]
            # x_text[0] = "Head: " + x_text[0]
            # x_text[1] = "Tail: " + x_text[1]  # flag the head and tail
            self.pyg_graph.x = torch.stack([kg_dataset.text_feats[i] for i in x_text], dim=0)
            self.pyg_graph.edge_attr_feat = torch.stack([kg_dataset.text_feats[i] for i in edge_attr_text], dim=0)
            self.label_embeddings = torch.stack([kg_dataset.text_feats[i] for i in label_text], dim=0)
            self.label_text = label_text
            
        self.node_attrs = [key for key, value in self.pyg_graph if self.pyg_graph.is_node_attr(key)]
        self.edge_attrs = [key for key, value in self.pyg_graph if self.pyg_graph.is_edge_attr(key) and key != "edge_index"]
  

    def sample_subgraph_around_node(self, node_idx_list):
        match = (self.pyg_graph.edge_index.unsqueeze(0) == node_idx_list.unsqueeze(1).unsqueeze(1)).any(1)
        return torch.tensor([ np.random.choice(m.nonzero()[:,0]) for m in match])

    def get_node_subgraph(self, node_idx): # refactor as __item__, add supernode in here
        node_list, edge_index, edge_id = self.neighbor_sampler.sample_node(node_idx)
        data = {}
        data['center_node_idx'] = node_idx
        data['edge_index'] = edge_index
        data['num_nodes'] = len(node_list)

        for key in self.node_attrs:
            data[key] = self.pyg_graph[key][node_list]
        if self.kg_dataset.disk_features is not None:
            # x_id is node_list - node idx is the same in the feature matrix and in the entire graph
            data["x"] = torch.from_numpy(self.kg_dataset.disk_features["node"][data["x_id"]]).float()
        for key in self.edge_attrs:
            data[key] = self.pyg_graph[key][edge_id]
            if self.kg_dataset.disk_features is not None and key == "edge_attr":
                edge_types = self.pyg_graph.edge_attr[edge_id]
                data[key] = torch.from_numpy(self.kg_dataset.disk_features["rel"][edge_types]).float()
        if "edge_attr_feat" in data:
            data["edge_attr"] = data["edge_attr_feat"]
            del data["edge_attr_feat"]

        graph = Data(**data)
        graph.edge_index = torch.cat([graph.edge_index, graph.edge_index.flip(0)], 1)
        graph.edge_attr = torch.cat([graph.edge_attr, graph.edge_attr], 0)
        return graph

    def add_node_pooling_supernode(self, data):
        for key in self.node_attrs:
            value = data[key]
            data[key] = torch.cat((value, torch.zeros(1, *value.shape[1:], dtype=value.dtype, layout=value.layout, device=value.device)))

        supernode_idx = data.num_nodes
        data.supernode = torch.tensor([supernode_idx])
        data.edge_index_supernode = torch.tensor([[0], [supernode_idx]], dtype=int)
        data.edge_index_from_supernode = torch.tensor([[supernode_idx], [0]], dtype=int)
        data.num_nodes += 1

    def get_subgraph(self, edge_idx): 
        rev = False
        if edge_idx < 0:
            edge_idx = - edge_idx - 1
            rev = True
        e = self.pyg_graph.edge_index[:,edge_idx].tolist()
        if rev:
            e = [e[1], e[0]]
        node_list, edge_index, edge_id = self.neighbor_sampler.sample_node(e)

        # remove the direct edge
        edge_index = edge_index[:, edge_id != edge_idx]
        edge_id = edge_id[edge_id != edge_idx]

        data = {}
        data['center_node_idx'] = e
        data['edge_index'] = edge_index
        data['num_nodes'] = len(node_list)
        for key in self.node_attrs:
            data[key] = self.pyg_graph[key][node_list]
        if self.kg_dataset.disk_features is not None:
            data["x"] = torch.from_numpy(self.kg_dataset.disk_features["node"][data["x_id"]]).float()
        for key in self.edge_attrs:
            data[key] = self.pyg_graph[key][edge_id]
            if self.kg_dataset.disk_features is not None and key == "edge_attr":
                edge_types = self.pyg_graph.edge_attr[edge_id]
                data[key] = torch.from_numpy(self.kg_dataset.disk_features["rel"][edge_types]).float()
        if "edge_attr_feat" in data:
            data["edge_attr"] = data["edge_attr_feat"]
            del data["edge_attr_feat"]
        
        
        graph = Data(**data)

        return graph


    def add_pooling_supernode(self, data):
        # for key in [key for key, value in data if data.is_node_attr(key)]:
        for key in list(set(self.node_attrs).union(set("x"))):
            value = data[key]
            data[key] = torch.cat((value, torch.zeros(1, *value.shape[1:], dtype=value.dtype, layout=value.layout, device=value.device)))

        supernode_idx = data.num_nodes
        data.supernode = torch.tensor([supernode_idx])
        data.edge_index_supernode = torch.tensor([[0, 1], [supernode_idx, supernode_idx]], dtype=int)
        data.edge_index_from_supernode = torch.tensor([[supernode_idx, supernode_idx], [0, 1]], dtype=int)
        data.num_nodes += 1

    def __getitem__(self, index):
        """
        Returns the subgraph at index and adds the supernode
        """
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]
        elif isinstance(index, tuple):
            return tuple(self.__getitem__(i) for i in index)
        elif isinstance(index, dict):
            return {key: self.__getitem__(value) for key, value in index.items()}
        elif not isinstance(index, int):
            return index

        assert index >= -len(self) and index < len(self)
        if self.node_graph:
            graph = self.get_node_subgraph(index)

            # Add supernode to the graph
            self.add_node_pooling_supernode(graph)

            return graph
        else:
            graph = self.get_subgraph(index)

            # Add supernode to the graph
            self.add_pooling_supernode(graph)

        return graph

    def __len__(self):
        return self.pyg_graph.edge_index.shape[1] 
