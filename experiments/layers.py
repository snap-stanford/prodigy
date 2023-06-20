import sys
import os


sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))


from models.metaGNN import MetaGNN, MetaTransformer, MetaTransformerPytorch, MetaAverage
from models.get_model import get_model, print_num_trainable_params
from models.supernode_propagation_layers import (
    BgGraphToSupernodePropagator,
    SupernodeToBgGraphPropagator,
    SupernodeToBgGraphGlobalPropagator,
    BgGraphToSupernodePropagatorCat,
    BgGraphToSupernodePropagatorPool,
)
from transformers import GPT2Model, GPT2Config

def get_module_list(module_string, emb_dim, edge_attr_dim, input_dim, dropout, reset_after_layer, attention_mask_scheme, has_final_back, msg_pos_only, batch_norm_metagraph=True, batch_norm_encoder=True, gnn_use_relu=False):
    '''
    The idea is that we describe the order in which different modules are applied with a simple comma-separated string.

    :param module_string: The comma-separated string describing the modules to be used.
    :param emb_dim:
    :param edge_attr_dim:
    :param input_dim:
    :return:
    '''

    module_list = []
    is_first_layer = True
    for layer in module_string.upper().split(","):
        if layer[0] == "S":
            # GraphSAGE background graph layer:
            # e. g. S = 1-layer GraphSAGE
            # e. g. S2 = 2-layer GraphSAGE
            n_layer = 1
            if layer[1:].isnumeric():
                n_layer = int(layer[1:])
            in_dim = emb_dim
            if is_first_layer:
                is_first_layer = False
                in_dim = input_dim  # the first layer takes the input features as input

            module_list.append(
                get_model(add_to_dim_in=0,
                          emb_dim=emb_dim,
                          n_layer=n_layer,
                          input_dim=in_dim,
                          classification_only=False,
                          gnn_type="sage",
                          edge_attr_dim=edge_attr_dim,
                          dropout = dropout,
                          reset_after_layer = reset_after_layer,
                          batch_norm=batch_norm_encoder)
                )
        elif layer[0] == "G":
            # GraphSAGE background graph layer:
            # e. g. S = 1-layer GraphSAGE
            # e. g. S2 = 2-layer GraphSAGE
            n_layer = 1
            if layer[1:].isnumeric():
                n_layer = int(layer[1:])
            in_dim = emb_dim
            if is_first_layer:
                is_first_layer = False
                in_dim = input_dim  # the first layer takes the input features as input

            module_list.append(
                get_model(add_to_dim_in=0,
                          emb_dim=emb_dim,
                          n_layer=n_layer,
                          input_dim=in_dim,
                          classification_only=False,
                          gnn_type="gat",
                          edge_attr_dim=edge_attr_dim,
                          dropout = dropout,
                          reset_after_layer = reset_after_layer,
                          batch_norm=batch_norm_encoder)
                )
        elif layer[0] == "M" and not layer.startswith("MX"):
            # meta layer
            n_layer = 1
            if layer[1:].isnumeric():
                n_layer = int(layer[1:])
            module_list.append(
                MetaGNN(emb_dim=emb_dim, edge_attr_dim=2, n_layers=n_layer, heads=8, dropout = dropout, has_final_back=has_final_back,
                        msg_pos_only=msg_pos_only, batch_norm=batch_norm_metagraph, gat_layer=False, use_relu=gnn_use_relu)
            )
        elif layer[0] == "W":
            # modified meta layer (experimental)
            n_layer = 1
            if layer[1:].isnumeric():
                n_layer = int(layer[1:])
            module_list.append(
                MetaGNN(emb_dim=emb_dim, edge_attr_dim=2, n_layers=n_layer, heads=8, dropout = dropout, has_final_back=has_final_back,
                        msg_pos_only=msg_pos_only, batch_norm=batch_norm_metagraph, gat_layer=True, use_relu=gnn_use_relu)
            )
        elif layer.startswith("MX"):
            # meta layer without self loops
            n_layer = 1
            if layer[2:].isnumeric():
                n_layer = int(layer[2:])
            module_list.append(
                MetaGNN(emb_dim=emb_dim, edge_attr_dim=2, n_layers=n_layer, heads=8, dropout = dropout, has_final_back=has_final_back,
                        msg_pos_only=msg_pos_only, self_loops=False, batch_norm=batch_norm_metagraph, use_relu=gnn_use_relu)

            )
        elif layer[0] == "A":
            # meta layer
            n_layer = 1
            if layer[1:].isnumeric():
                n_layer = int(layer[1:])
            module_list.append(
                MetaAverage(emb_dim=emb_dim, edge_attr_dim=2, n_layers=n_layer, heads=8, dropout = dropout)
            )
        elif layer[0] == "T":
            # meta layer
            n_layer = 1
            if layer[1:].isnumeric():
                n_layer = int(layer[1:])

            module_list.append(
                MetaTransformerPytorch(
                    GPT2Config(
                        vocab_size = 0,
                        n_positions=1024,
                        n_embd=emb_dim,
                        n_layer=n_layer,
                        n_head=4,
                    ),
                    attention_mask_scheme
                )
            )
        elif layer[0] == "P":
            # meta layer
            n_layer = 1
            if layer[1:].isnumeric():
                n_layer = int(layer[1:])

            module_list.append(
                MetaTransformer(
                    GPT2Model,
                    GPT2Config(
                        vocab_size = 0,
                        n_positions=1024,
                        n_embd=emb_dim,
                        n_layer=n_layer,
                        n_head=4,
                    )
                )
            )
        elif layer.upper() == "UX":
            # Up : aggregation from background graph to supernode
            module_list.append(
                BgGraphToSupernodePropagatorCat(emb_dim)
            )
        elif layer.upper() == "UY":
            # Up : aggregation from background graph to supernode
            module_list.append(
                BgGraphToSupernodePropagatorPool(emb_dim)
            )

        elif layer[0] == "U":
            # Up : aggregation from background graph to supernode
            module_list.append(
                BgGraphToSupernodePropagator()
            )
        elif layer.upper() == "D+ATT":
            # Down + attention: whole subgraph-level attention as well as aggregation from supernode to background graph
            module_list.append(
                SupernodeToBgGraphGlobalPropagator(emb_dim=emb_dim)
            )
        elif layer[0] == "D":
            # Down : aggregation from supernode to background graph
            module_list.append(
                SupernodeToBgGraphPropagator(emb_dim=emb_dim)
            )
        else:
            raise NotImplementedError

    return module_list

