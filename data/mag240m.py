import os
import random
from collections import defaultdict
import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from experiments.sampler import NeighborSamplerCacheAdj
from ogb.lsc import MAG240MDataset
from .dataset import SubgraphDataset
from .dataloader import NeighborTask, MultiTaskSplitWay, MultiTaskSplitBatch, MulticlassTask, ParamSampler, BatchSampler, Collator, ContrastiveTask
from .augment import get_aug

class MAG240MSubgraphDataset(SubgraphDataset):
    def get_subgraph(self, *args, **kwargs):
        graph = super().get_subgraph(*args, **kwargs)
        graph.x = graph.x.float()  # it was half
        return graph


def get_mag240m_dataset(root, n_hop=2, **kwargs):
    dataset = MAG240MDataset(root)

    # Check if "mag240m_fts_adj_label.pt" exists, otherwise load
    # "mag240m_fts_adj.pt" and save it with labels.
    do_load = False
    adj_bi_cached = os.path.exists(os.path.join(root, "mag240m_adj_bi.pt"))
    if do_load and os.path.exists(os.path.join(root, "mag240m_fts_adj_label.pt")):
        print("Loading MAG240M dataset from .pt...")
        edge_index, fts, num_paper = torch.load(os.path.join(root, "mag240m_fts_adj_label.pt"))
    else:
        print("Loading MAG240M dataset...")
        t = time.time()
        edge_index = None
        if not adj_bi_cached:
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
        fts = torch.from_numpy(dataset.paper_feat)
        num_paper = dataset.num_papers
        data = (edge_index, fts, dataset.num_papers)
        print(f"Loading MAG240M dataset takes {time.time() - t:.2f} seconds")
        if do_load:
            torch.save(data, os.path.join(root, "mag240m_fts_adj_label.pt"))
    print("Done loading MAG240M dataset.")
    graph_ns = None if adj_bi_cached else Data(edge_index=edge_index, num_nodes=num_paper)
    neighbor_sampler = NeighborSamplerCacheAdj(os.path.join(root, "mag240m_adj_bi.pt"), graph_ns, n_hop)
    print("Done loading MAG240M neighbor sampler.")
    graph = Data(x=fts, num_nodes=num_paper)
    return MAG240MSubgraphDataset(graph, neighbor_sampler)


def mag240m_labels(split, node_split = "", root="dataset", remove_cs=True):
    dataset = MAG240MDataset(root)
    num_classes = dataset.num_classes

    if remove_cs:
        arxiv_labels = [
            0, 1, 3, 6, 9, 16, 17, 23, 24, 26,
            29, 39, 42, 47, 52, 57, 59, 63, 73, 77,
            79, 85, 86, 89, 94, 95, 105, 109, 114, 119,
            120, 122, 124, 130, 135, 137, 139, 147, 149, 152]
        labels = list(set(range(num_classes)) - set(arxiv_labels))
        additional = arxiv_labels
    else:
        labels = list(range(num_classes))
        additional = []
    generator = random.Random(42)
    generator.shuffle(labels)

    test_val_length = 5
    TEST_LABELS = labels[:test_val_length] + additional
    VAL_LABELS = labels[test_val_length: test_val_length * 2] + additional # Hacky but not trivial to fix uneven sampling with random sampling for high way (like 30)
    TRAIN_LABELS = labels[test_val_length * 2:]

    # TRAIN_LABELS = [88, 2, 125, 54, 76, 38, 121, 145, 61, 112, 64, 3, 94, 52, 32, 83, 14,
    #     140, 63, 135, 124, 91, 109, 111, 86, 106, 95, 16, 113, 66, 53, 25, 74, 75, 60, 98,
    #     101, 133, 36, 85, 120, 65, 17, 51, 137, 4, 89, 141, 41, 152, 78, 127, 138, 82, 31,
    #     134, 21, 9, 34, 146, 116, 47, 20, 99, 81, 115, 126, 105, 117, 92, 104, 102, 29, 84,
    #     110, 142, 90, 24, 73, 46, 79, 80, 37, 150, 10, 118, 15, 68, 58, 93, 5, 103, 33, 77,
    #     44, 128, 45, 12, 48, 11, 13, 43, 97, 122, 27, 19, 147, 87, 143, 40, 1, 71, 114, 56,
    #     107, 50, 151, 129, 59, 55, 23, 7, 8, 108, 22, 139, 26, 35, 57, 62, 70, 6, 28]
    # VAL_LABELS = [149, 18, 130, 119, 96, 0, 132, 42, 72, 30]
    # TEST_LABELS = [136, 131, 148, 39, 67, 49, 123, 144, 100, 69]

    label = dataset.all_paper_label
    if split == "train":
        label_set = set(TRAIN_LABELS)
    elif split == "val":
        label_set = set(VAL_LABELS)
    elif split == "test":
        if not remove_cs:
            print("Warning: remove_cs is set to false, might not be enough samples.")
        label_set = set(TEST_LABELS)
    else:
        raise ValueError(f"Invalid split: {split}")

    return label, label_set, num_classes


def get_mag240m_dataloader(dataset, task_name, split, node_split, batch_size, n_way, n_shot, n_query, batch_count, root, num_workers, aug, aug_test, **kwargs):
    seed = sum(ord(c) for c in split)
    if split == "train" or aug_test:
        aug = get_aug(aug, dataset.graph.x)
    else:
        aug = get_aug("")
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
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 2
        sampler = BatchSampler(
            batch_count,
            NeighborTask(neighbor_sampler, len(dataset), "inout"),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(len(dataset), -1)
    elif task_name == "classification":
        labels, label_set, num_classes = mag240m_labels(split, root=root, remove_cs=True)
        sampler = BatchSampler(
            batch_count,
            MulticlassTask(labels, label_set),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(num_classes, -1)
    # Classification and neighbor matching - multitask splitbatch
    elif task_name.startswith("cls_nm"):
        labels, label_set, num_classes = mag240m_labels(split, root=root)
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 2
        
        if task_name.endswith("sb"):
            task_base = MultiTaskSplitBatch([
                MulticlassTask(labels, label_set),
                NeighborTask(neighbor_sampler, len(dataset), "inout")
            ], ["mct", "nt"], [1, 3])
        elif task_name.endswith("sw"):
            task_base = MultiTaskSplitWay([
                MulticlassTask(labels, label_set), 
                NeighborTask(neighbor_sampler, len(dataset), "inout")
            ], ["mct", "nt"], split="even")
        
        sampler = BatchSampler(
            batch_count,
            task_base,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = {}
        label_meta["mct"] = torch.zeros(1, 768).expand(num_classes, -1)
        label_meta["nt"] = torch.zeros(1, 768).expand(len(dataset), -1)
    else:
        raise ValueError(f"Unknown task for MAG240M: {task_name}")
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=Collator(label_meta, aug=aug))
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
