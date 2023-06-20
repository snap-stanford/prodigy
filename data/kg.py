import os
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib
from data.load_kg_dataset import SubgraphFewshotDatasetWithTextFeats
from .dataset import KGSubgraphDataset
from .dataloader import KGNeighborTask, MulticlassTask, ParamSampler, BatchSampler, KGCollator, ContrastiveTask, NeighborTask, Collator, MultiTaskSplitBatch, MultiTaskSplitWay
from .augment import get_aug
from torch_geometric.data import Data
from experiments.sampler import NeighborSamplerCacheAdj
import copy
import random
import numpy as np
import json

def get_csr_split(root, name):
    # get CSR label split for the given dataset
    result = {}
    for subset in ["pretrain", "dev", "test"]:
        fname = subset + "_tasks"
        fname += ".json"
        fname = os.path.join(root, name, fname)
        if subset not in  result:
            result[subset] = set()
        if os.path.exists(fname):
            #with open(fname) as f:
            #    result[subset] = set(json.load(f)).union(result[subset])
            result[subset] = set(list(json.load(open(fname)).keys())).union(result[subset])
    return result


def get_kg_dataset(root, name, n_hop=2, bert=None, bert_device="cpu", **kwargs):
    assert name in ["NELL", "FB15K-237", "ConceptNet", "Wiki", "WikiKG90M"]  

    kind = "union"
    sampler_type = "new"
    subset = "test"
    hop = 2
    shot = 3

    if name == "ConceptNet":
        hop = 1
    if name == "NELL":
        hop = 1
    if name == "FB15K-237":
        hop = 1
    pretrained_embeddings = None 
    dataset = SubgraphFewshotDatasetWithTextFeats(root=root, dataset=name, mode=subset, hop=hop, kind = kind, shot=shot, preprocess=False,
                     bert=bert, device=bert_device, embeddings_model=pretrained_embeddings, graph_only = True)

    graph_ns = Data(edge_index=dataset.graph.edge_index, num_nodes=dataset.graph.num_nodes)
    neighbor_sampler = NeighborSamplerCacheAdj(os.path.join(root, name, f"{name}_adj.pt"), graph_ns, hop)
    dataset.csr_split = get_csr_split(root, name)
    return KGSubgraphDataset(dataset, neighbor_sampler, sampler_type, node_graph = kwargs["node_graph"])


def idx_split(n, fracs=[0.7, 0.1, 0.2]):
    generator = random.Random(42)
    labels = list(range(n))
    generator.shuffle(labels)
    i = int(n * fracs[0])
    j = int(n * (fracs[0] + fracs[1]))
    train = labels[:i]
    val = labels[i:j]
    test = labels[j:]
    return {"train": train, "valid": val, "test": test}


def kg_labels(dataset, split, node_split = "", all_test=False, csr_split=False):
    num_classes = dataset.pyg_graph.edge_attr.max().item() +1
    print(num_classes)
    labels = list(range(num_classes))
    generator = random.Random(42)
    generator.shuffle(labels)
    #TEST_LABELS = labels[:50]
    #VAL_LABELS = labels[50: 100]
    #TRAIN_LABELS = labels[100:]
    if num_classes <= 20:
        # ConceptNet
        i = int(num_classes / 3)
        j = int(num_classes * 2/3)
    else:
        # FB and NELL
        i = int(num_classes * 0.6)
        j = int(num_classes * 0.8)

    TEST_LABELS = labels[:i]
    VAL_LABELS = labels[i: j]
    TRAIN_LABELS = labels[j:]
    if csr_split:
        train_tasks, test_tasks, val_tasks = dataset.kg_dataset.csr_split["pretrain"], dataset.kg_dataset.csr_split["test"], dataset.kg_dataset.csr_split["dev"]
        assert train_tasks.intersection(test_tasks) == set() and train_tasks.intersection(val_tasks) == set() and test_tasks.intersection(val_tasks) == set()
        TRAIN_LABELS = [dataset.label_text.index(task) for task in train_tasks if task in dataset.label_text]
        VAL_LABELS = [dataset.label_text.index(task) for task in val_tasks  if task in dataset.label_text]
        TEST_LABELS = [dataset.label_text.index(task) for task in test_tasks  if task in dataset.label_text]
    if all_test:
        TEST_LABELS = labels
        VAL_LABELS = labels
        TRAIN_LABELS = labels
        print("Setting all labels for evaluation...")
    else:
        print("TEST_LABELS", len(TEST_LABELS))
        print("VAL_LABELS", len(VAL_LABELS))
        print("TRAIN_LABELS", len(TRAIN_LABELS))
        #print("i=", i, "j=",j)

    # TRAIN_LABELS = [88, 2, 125, 54, 76, 38, 121, 145, 61, 112, 64, 3, 94, 52, 32, 83, 14,
    #     140, 63, 135, 124, 91, 109, 111, 86, 106, 95, 16, 113, 66, 53, 25, 74, 75, 60, 98,
    #     101, 133, 36, 85, 120, 65, 17, 51, 137, 4, 89, 141, 41, 152, 78, 127, 138, 82, 31,
    #     134, 21, 9, 34, 146, 116, 47, 20, 99, 81, 115, 126, 105, 117, 92, 104, 102, 29, 84,
    #     110, 142, 90, 24, 73, 46, 79, 80, 37, 150, 10, 118, 15, 68, 58, 93, 5, 103, 33, 77,
    #     44, 128, 45, 12, 48, 11, 13, 43, 97, 122, 27, 19, 147, 87, 143, 40, 1, 71, 114, 56,
    #     107, 50, 151, 129, 59, 55, 23, 7, 8, 108, 22, 139, 26, 35, 57, 62, 70, 6, 28]
    # VAL_LABELS = [149, 18, 130, 119, 96, 0, 132, 42, 72, 30]
    # TEST_LABELS = [136, 131, 148, 39, 67, 49, 123, 144, 100, 69]
    label = dataset.pyg_graph.edge_attr
    if split == "train":
        label_set = set(TRAIN_LABELS)
    elif split == "val":
        label_set = set(VAL_LABELS)
    elif split == "test":
        label_set = set(TEST_LABELS)
    else:
        raise ValueError(f"Invalid split: {split}")

    return label, label_set, num_classes

def kg_task_no_labels_split(labels, dataset, label_set, linear_probe, train_cap=3, split="train"):
    # labels = edge_attr
    edge_index = dataset.pyg_graph.edge_index
    rnd_split = idx_split(len(edge_index[0]))
    train_label = labels.numpy().copy()
    split_idx = np.array(rnd_split["train"])
    train_label[split_idx] = -1 - train_label[split_idx]
    train_label = -1 - train_label
    COUNT_CAP = train_cap
    label = train_label
    #if label_set is None:
    #    label_set = list(range(max(labels) + 1))
    if COUNT_CAP is not None:
        for i in range(max(labels) + 1):
            idx = (train_label == i)
            if idx.sum() > COUNT_CAP:
                disabled_idx = np.where(idx)[0][COUNT_CAP:]
                train_label[disabled_idx] = -1 - i
    if split == "train":
        label = train_label
        train_label = None
    else:
        split_idx = np.array(rnd_split[split if split != "val" else "valid"])
        label[split_idx] = -1 - label[split_idx]
        label = -1 - label
    return MulticlassTask(label, label_set, train_label, linear_probe=linear_probe)


def get_kg_dataloader(dataset, task_name, split, node_split, batch_size, n_way, n_shot, n_query, batch_count, root, num_workers, aug, aug_test, train_cap, linear_probe, label_set=None, all_test=False, **kwargs):
    # NELL label set example 111 92 73 152 206 184 167 65 91 97 239 170 120 155 76 181 256 194 126 217 105 154 118 94 178 106 12 251 195 243 99 250 252 115 9 82 158 216 64 22 48 247 224 219 77 84 0 78 70 51
    seed = sum(ord(c) for c in split)
    #seed = None
    # if type(dataset) == tuple:
    #     if split == "train":
    #         dataset = dataset[0]
    #     elif split == "test":
    #         dataset = dataset[2]
    #     elif split in ["val", "valid", "dev", "validation"]:
    #         dataset = dataset[1]
    if "split_labels" in kwargs:
        split_labels = kwargs["split_labels"]
    else:
        split_labels = True
    if "csr_split" in kwargs:
        csr_split = kwargs["csr_split"]
    else:
        csr_split = False
    if split == "train" or aug_test:
        aug = get_aug(aug, dataset.pyg_graph.x)
    else:
        aug = get_aug("")
    is_multiway = True
    if n_way == 1:
        n_way = n_shot + n_query + 1
        is_multiway = False
    if task_name == "same_graph":
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 0
        sampler = BatchSampler(
            batch_count,
            ContrastiveTask(len(dataset)),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(len(dataset), -1)
    elif task_name == "neighbor_matching":
        num_nodes = dataset.pyg_graph.num_nodes
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 1
        sampler = BatchSampler(
            batch_count,
            KGNeighborTask(dataset, neighbor_sampler, num_nodes, "inout", is_multiway),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(num_nodes, -1)
    elif task_name == "sn_neighbor_matching":
        num_nodes = dataset.pyg_graph.num_nodes
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 1
        sampler = BatchSampler(
            batch_count,
            NeighborTask(neighbor_sampler, num_nodes, "inout"),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(num_nodes, -1)
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=Collator(label_meta, aug=aug, is_multiway=is_multiway))
        return dataloader

    elif task_name == "multiway_classification":
        labels, label_set_split_lbls, num_classes = kg_labels(dataset, split, node_split, all_test, csr_split)
        if split_labels:
            task = MulticlassTask(labels, label_set_split_lbls, train_label=None, linear_probe=linear_probe)
        else:
            assert label_set is not None, "label_set must be provided for no_split_labels"
            task = kg_task_no_labels_split(labels, dataset=dataset, train_cap=train_cap, split=split, label_set=label_set, linear_probe=linear_probe) 
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )

        label_meta = torch.clone(dataset.label_embeddings)
    elif task_name == "cls_nm":
        labels, label_set, num_classes = kg_labels(dataset, split, node_split, all_test, csr_split)
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 1
        num_nodes = dataset.pyg_graph.num_nodes
        if task_name.endswith("sw"):
            task_base = MultiTaskSplitWay([
                MulticlassTask(labels, label_set),
                KGNeighborTask(dataset, neighbor_sampler, num_nodes, "inout", is_multiway),
            ], ["mct", "nt"], split="even")
        else:
            task_base = MultiTaskSplitBatch([
                    MulticlassTask(labels, label_set),
                    KGNeighborTask(dataset, neighbor_sampler, num_nodes, "inout", is_multiway)
                ], ["mct", "nt"], [98, 2])
        sampler = BatchSampler(
            batch_count,
            task_base,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = {}
        label_meta["mct"] = torch.zeros(1, 768).expand(num_classes, -1)
        label_meta["nt"] = torch.zeros(1, 768).expand(num_nodes, -1)
    else:
        raise ValueError(f"Unknown task for KG: {task_name}")
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=KGCollator(label_meta, aug=aug, is_multiway=is_multiway))
    return dataloader


if __name__ == "__main__":
    from tqdm import tqdm
    import cProfile

    root = "../FSdatasets/mag240m"
    n_hop = 2

    dataset = get_mag240m_dataset(root, n_hop)
    dataloader = get_mag240m_dataloader(dataset, "train", "", 5, 3, 3, 24, 10000, root, 10)

    for batch in tqdm(dataloader):
        pass
