import torch
import argparse

import sys
import os
from datetime import datetime
sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))


def get_params():
    args = argparse.ArgumentParser()

    args.add_argument("-root", "--root", default="./FSdatasets", type=str)
    args.add_argument("-dataset", "--dataset", default="arxiv", type=str)
    args.add_argument("-invalidate_cache", "--invalidate_cache", default=False, type=bool)
    # if true, it will regenerate preprocessed cache
    args.add_argument("-ds_cap", "--dataset_len_cap", default=10000, type=int)
    args.add_argument("-val_cap", "--val_len_cap", default=None, type=int)
    args.add_argument("-test_cap", "--test_len_cap", default=None, type=int)
    args.add_argument("-shuffle_index", "--shuffle_index", default=False, type=bool) # For KG datasets, shuffle index to get a different task each time if using ds_cap 1
    # will cap length of training and testing datasets (use this for debugging or when using smaller GPU)
    args.add_argument("-force_cache", "--force_cache", default=False, type=bool)  # will use preprocessed cache
    args.add_argument("-cl_only", "--classification_only", default=False, type=bool) # only set this to true when using the very basic arxiv dataset!!! (this is basic node classification where labels are the same in train and test)
    args.add_argument("-esp", "--early_stopping_patience", default=20, type=int) # early stopping patience (in validation epochs, so with default eval_epoch argument 20 * 10 = 200 epochs)
    args.add_argument("--reset_after_layer", default=None, nargs='+', type=int)
    args.add_argument("-original_features", "--original_features", default=False, type=bool)
    args.add_argument("-override_log", "--override_log", default=False, type=bool)


    args.add_argument("-seed", "--seed", default=None, type=int)

    args.add_argument("-metric", "--metric", default="Acc", choices=["Acc"])

    # Training-specific params
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-epochs", "--epochs", default=12, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=10, type=int)  # deprecated
    args.add_argument("-eval_epo", "--eval_epoch", default=10, type=int)  # deprecated
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=10, type=int)  # deprecated
    args.add_argument("-prt_step", "--print_step", default=2000, type=int)
    args.add_argument("-eval_step", "--eval_step", default=2000, type=int)
    args.add_argument("-ckpt_step", "--checkpoint_step", default=2000, type=int)
    args.add_argument("-bs", "--batch_size", default=5, type=int) 
    args.add_argument("-weight_decay", "--weight_decay", default=0.001, type=float)
    args.add_argument("-dropout", "--dropout", default=0, type=float)
    args.add_argument("-txt_dropout", "--text_features_dropout", default=0, type=float)  # additionally drop out text features
    args.add_argument("-rel_sample_seed", "--rel_sample_random_seed", default=None, type=float)  # seed for sampling relations

    args.add_argument("-split_train_nodes", "--split_train_nodes", default=False, type=bool) # Split train nodes into 'train' and 'val'

    args.add_argument("-verbose", "--verbose", default=False, type=bool)

    args.add_argument("-workers", "--workers", default=10, type=int)  # Number of workers per dataloader
    args.add_argument("-gpu", "--device", default=123, type=int)  # device 123 means CPU


    # GNN- and model-specific parameters
    args.add_argument("-input_dim", "--input_dim", default=768, type=int)  # this is bert dim etc.
    args.add_argument("-emb_dim", "--emb_dim", default=256, type=int)
    args.add_argument("-gnn_type", "--gnn_type", default="sage", type=str)  # support "gin", "no_msg_passing", "sage"
    args.add_argument("-n_layer", "--n_layer", default=1, type=int)
    args.add_argument("-meta_n_layer", "--meta_n_layer", default=1, type=int)
    args.add_argument("-gnn2", "--second_gnn", default="Atten", type=str)  # "vanilla" or "gat"
    args.add_argument("--attention_mask_scheme", default="causal", type=str)  # the name of the pretraining task
    args.add_argument("-skip", "--skip_path", default=False, type=bool)
    args.add_argument("-has_final_back", "--has_final_back", default=False, type=bool)

    args.add_argument("-layers", "--layers", default="S,U,M", type=str)  # default: GraphSAGE->Up->Metagraph (see experiments/layers.py for more info)
    args.add_argument("-ignore_label_embs", "--ignore_label_embeddings", default=True, type=bool)
    args.add_argument("-zero_lbl", "--zero_label_embeddings", default=False, type=bool)
    args.add_argument("-not_freeze_learned_label_embedding", "--not_freeze_learned_label_embedding", default=False, type=bool)
    args.add_argument("-linear_probe", "--linear_probe", default=False, type=bool)
    args.add_argument("-fdf", "--fix_datasets_first", default=False,
                      type=bool)  # Whether to convert datasets to list first (no sampling involved later).
    # This should generally not be used as the resulting files would be way too large

    args.add_argument("-no_bn_metagraph", "--no_bn_metagraph", default=False,  # no batch norm metagraph
                      type=bool)
    args.add_argument("-no_bn_encoder", "--no_bn_encoder", default=False,  # no batch norm on S layers etc.
                      type=bool)

    args.add_argument("-calc_ranks", "--calc_ranks", default=False, type=bool)  # Whether to calc MRR and HITS ranks.
    args.add_argument("-eval_only", "--eval_only", default=False, type=bool)  # Eval. only mode (no training, only one pass of testing ds at the beginning and then quit)
    args.add_argument("-meta_pos", "--meta_gnn_pos_only", default=False, type=bool)  # Whether to use only positive edges for meta graph

    ###  Few-shot task parameters  ###
    args.add_argument("-task", "--task_name", default="classification", type=str)  # the name of the pretraining task
    args.add_argument("-zeroshot", "--zero_shot", default=False, type=bool) # if True, messages will NOT be passed along the metagraph edges.
    args.add_argument("-no_split_labels", "--no_split_labels", default=True, type=bool) # split train/val/test with original dataset split
    args.add_argument("-all_test", "--all_test", default=False,
                      type=bool)  # Set train/test/val labels to the same label set (for testing purposes)
    args.add_argument("-train_cap", "--train_cap", default=None, type=int) # split train/val/test with original dataset split
    args.add_argument('--label_set', type=str, nargs='+')
    args.add_argument("-csr_split", "--csr_split", default=False, type=bool)  # Whether to use CSR split...

    args.add_argument("-way", "--n_way", default=3, type=int) # how many labels do we want in each few-shot task
    args.add_argument("-shot", "--n_shots", default=3, type=int) # if not zeroshot, how many shots do we want in the training dataset?
    args.add_argument("-qry", "--n_query", default=24, type=int)

    args.add_argument("-way_u", "--n_way_upper", default=-1, type=int) # If defined, will set the upper bound for n_way
    args.add_argument("-shot_u", "--n_shots_upper", default=-1, type=int) # If defined, will set the upper bound for n_shots
    args.add_argument("-qry_u", "--n_query_upper", default=-1, type=int) # If defined, will set the upper bound for n_query
    args.add_argument("-max_length", default=-1, type=int) 

    ### data augmentation parameters ###
    args.add_argument("-aug", "--augmentation", default="", type=str)
    args.add_argument("-aug_test", "--augment_test", default=False, type=bool)  # if set, the valid set and the test set are also augmented
    args.add_argument("-attr", "--attr_regression_weight", default=0.0, type=float)


    args.add_argument("-prefix", "--prefix", default="exp1", type=str) # prefix for the experiment name for wandb
    args.add_argument("-timestamp", "--timestamp", default=None, type=str)

    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)

    args.add_argument("-bert", "--bert_emb_model", default="multi-qa-distilbert-cos-v1")
    args.add_argument("-bert_cache", "--bert_cache", default="multi-qa-distilbert-cos-v1")

    args.add_argument("-kg_emb", "--kg_emb_model", default="", type=str)   #  "TransE", "ComplEx", etc.
    args.add_argument("-pretrained", "--pretrained_model_run", default="", type=str)
    #  Name of WanDB run to pull the best model from.

    args.add_argument("-smalldataset", "--small_dataset", default=False,
                      type=bool)  # use for debugging  - very small dataset

    args.add_argument("-exptype", "--experiment_type", default="metagraph")


    args = args.parse_args()

    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.device == 123:
        params["device"] = torch.device('cpu')
    else:
        params['device'] = torch.device('cuda:' + str(args.device))
    if params["timestamp"] is None:
        params["timestamp"] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    params["exp_name"] = params["prefix"] + "_" + params["timestamp"]

    return params

