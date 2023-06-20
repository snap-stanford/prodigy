import numpy as np
import random
import torch

torch.multiprocessing.set_sharing_strategy("file_system") 

import sys
import os
torch.autograd.set_detect_anomaly(True)

sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))

from experiments.params import get_params
from experiments.trainer import TrainerFS

from data.data_loader_wrapper import get_dataset_wrap

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    torch.set_num_threads(4)

    params = get_params()
    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    if params["dataset"] in ["FB15K-237", "NELL", "ConceptNet", "Wiki"]:
        print("Using KG dataset - setting language model to sentence-transformers/all-mpnet-base-v2")
        params["bert_emb_model"] = "sentence-transformers/all-mpnet-base-v2"
    datasets = get_dataset_wrap(
        root=params["root"],
        dataset=params["dataset"],
        force_cache=params["force_cache"],
        small_dataset=params["small_dataset"],
        invalidate_cache=None,
        original_features=params["original_features"],
        n_shot=params["n_shots"],
        n_query=params["n_query"],
        bert=None if params["original_features"] else params["bert_emb_model"],
        bert_device=params["device"],
        val_len_cap=params["val_len_cap"],
        test_len_cap=params["test_len_cap"],
        dataset_len_cap=params["dataset_len_cap"],
        n_way=params["n_way"],
        rel_sample_rand_seed=params["rel_sample_random_seed"],
        calc_ranks=params["calc_ranks"],
        kg_emb_model=params["kg_emb_model"] if params["kg_emb_model"] != "" else None,
        task_name = params["task_name"],
        shuffle_index=params["shuffle_index"],
        node_graph = params["task_name"] == "sn_neighbor_matching"
    )

    trnr = TrainerFS(datasets,  params)

    trnr.train()



