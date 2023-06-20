import torch
import shutil
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import time
import numpy as np
import transformers
import pickle
import wandb
import sys
import os

sys.path.extend(os.path.join(os.path.dirname(__file__), "../"))

from models.gnn_with_edge_attr import gnn_models
from models.multilayer_gnn import MultiLayerGNN


def print_num_trainable_params(torch_module):
    model_parameters = filter(lambda p: p.requires_grad, torch_module.parameters())
    params = int(sum([np.prod(p.size()) for p in model_parameters]))
    print("Number of trainable parameters of the model:", params)
    return params



def get_model(emb_dim=128, bert_dim=768, n_layer=1, input_dim=768, edge_attr_dim=4, classification_only=False,
              gnn_type="gin", add_to_dim_in=None, dropout = 0, reset_after_layer = None, batch_norm=True):
    add_to_dim = 0  # add 1 to input dim because we add supernode feature at the beginning!
    GNNConv = gnn_models[gnn_type]
    if not classification_only:
        add_to_dim = 1
    if add_to_dim_in is not None:
        add_to_dim = add_to_dim_in
    if classification_only:
        edge_attr_dim = None
    if gnn_type == "gat":
        gnn1 = GNNConv(in_channels=input_dim + add_to_dim, edge_dim=edge_attr_dim, out_channels=emb_dim)
    else:
        gnn1 = GNNConv(x_dim=input_dim + add_to_dim, edge_attr_dim=edge_attr_dim, emb_dim=emb_dim, dropout = dropout, aggr="mean", batch_norm=batch_norm)
    
    all_gnns = [gnn1]
    assert n_layer >= 1
    for i in range(n_layer - 1):
        if gnn_type == "gat":
            all_gnns.append(GNNConv(in_channels=emb_dim, edge_dim=edge_attr_dim, out_channels=emb_dim))
        else:
            all_gnns.append(GNNConv(x_dim=emb_dim, edge_attr_dim=edge_attr_dim, emb_dim=emb_dim, aggr="mean", batch_norm=batch_norm))
        
    all_gnns = torch.nn.ModuleList(all_gnns)
    gnn_all = MultiLayerGNN(all_gnns, supernode_gnn=None, emb_dim=emb_dim, reset_after_layer=reset_after_layer)
    return gnn_all
