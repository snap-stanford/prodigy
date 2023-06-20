import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from ogb.nodeproppred import PygNodePropPredDataset
from .process_arxiv_categories import arxiv_cs_taxonomy
from experiments.sampler import NeighborSampler
from .dataset import SubgraphDataset
from .dataloader import MulticlassTask, ParamSampler, BatchSampler, Collator
from .augment import get_aug


def get_arxiv_dataset(root, n_hop=2, bert=None, bert_device="cpu", **kwargs):
    if bert is None:
        dataset = PygNodePropPredDataset("ogbn-arxiv", root=root)
        graph = dataset[0]
    else:
        cache_path = os.path.join(root, f"arxiv_text_{bert}.pt")
        if os.path.exists(cache_path):
            graph = torch.load(cache_path)
        else:
            graph = preprocess_arxiv_text_bert(root, bert, bert_device)
            torch.save(graph, cache_path)

    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)

    return SubgraphDataset(graph, neighbor_sampler)


def arxiv_task(split, node_split="", split_labels=True, train_cap = 3, label_set = range(40), linear_probe=False, ogb_root="dataset"):
    assert not node_split or split_labels

    dataset = PygNodePropPredDataset("ogbn-arxiv", root=ogb_root)
    graph = dataset[0]
    label = graph.y.squeeze(1).numpy().copy()
    if not split_labels:
        train_label = graph.y.squeeze(1).numpy().copy()
        split_idx = dataset.get_idx_split()["train"].numpy()
        train_label[split_idx] = -1 - train_label[split_idx]
        train_label = -1 - train_label
        # label_set = [0,1,2]
        # label_set = [10, 11, 14]
        COUNT_CAP = train_cap
        if COUNT_CAP is not None:
            # This only matters if we finetuning
            for i in range(40):
                idx = (train_label == i)
                if idx.sum() > COUNT_CAP:
                    disabled_idx = np.where(idx)[0][COUNT_CAP:]
                    train_label[disabled_idx] = -1 - i
        if split == "train": 
            label = train_label
            train_label = None
        else:
            # if split == "val":
            #     label_set = set(VAL_LABELS)
            # elif split == "test":
            #     label_set = set(TEST_LABELS)
            # else:
            #     raise ValueError(f"Invalid split: {split}")
            split_idx = dataset.get_idx_split()[split if split != "val" else "valid"].numpy()
            label[split_idx] = -1 - label[split_idx]
            label = -1 - label
        
    else:
        # Meta learning setting

        # Label split by G-Meta
        TRAIN_LABELS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 15, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 39]
        VAL_LABELS = [6, 12, 16, 19, 30, 35, 38]
        TEST_LABELS = [10, 11, 14, 21, 34, 36, 37]

        # Label split by TENT
        # TRAIN_LABELS = [32, 34, 3, 35, 38, 39, 10, 13, 16, 17, 18, 21, 23, 26, 27]
        # VAL_LABELS = [8, 12, 9, 1, 33]
        # TEST_LABELS = [28, 7, 0, 5, 2, 36, 6, 22, 15, 37, 30, 25, 29, 11, 20, 19, 31, 24, 14, 4]
        
        train_label = None
        if split == "train":
            label_set = set(TRAIN_LABELS)
        elif split == "val":
            label_set = set(VAL_LABELS)
        elif split == "test":
            label_set = set(TEST_LABELS)
        else:
            raise ValueError(f"Invalid split: {split}")

    return MulticlassTask(label, label_set, train_label, linear_probe)


def get_arxiv_dataloader(dataset, split, node_split, batch_size, n_way, n_shot, n_query, batch_count, root, bert, num_workers, aug, aug_test, split_labels, train_cap, linear_probe, label_set = range(40), **kwargs):
    mapping_file = os.path.join(root, "ogbn_arxiv", "mapping", "labelidx2arxivcategeory.csv.gz")
    arxiv_categ_vals = pd.merge(pd.read_csv(mapping_file), arxiv_cs_taxonomy, left_on="arxiv category",
                                                                right_on="id")
    arxiv_categ_vals = list(arxiv_categ_vals["name"].values)
    label_embeddings = bert.get_sentence_embeddings(arxiv_categ_vals)
    # Zeros like itself
    # label_embeddings = torch.zeros_like(label_embeddings)
    sampler = BatchSampler(
        batch_count,
        arxiv_task(split, node_split, split_labels, train_cap, label_set, linear_probe, root),
        ParamSampler(batch_size, n_way, n_shot, n_query, 1),
        seed=42,
    )
    if split == "train" or aug_test:
        aug = get_aug(aug, dataset.graph.x)
    else:
        aug = get_aug("")
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=Collator(label_embeddings, aug=aug))
    return dataloader


def preprocess_arxiv_text_bert(root, model_name, device):
    print("Preprocessing text features")
    dataset = PygNodePropPredDataset("ogbn-arxiv", root=root)
    graph = dataset[0]

    nodeidx2paperid = pd.read_csv(os.path.join(root, 'ogbn_arxiv', 'mapping', 'nodeidx2paperid.csv.gz'), index_col='node idx')
    titleabs_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv"
    titleabs = pd.read_csv(titleabs_url, sep='\t', names=['paper id', 'title', 'abstract'], index_col='paper id')
    titleabs = nodeidx2paperid.join(titleabs, on='paper id')

    text = titleabs["title"] + ". " + titleabs["abstract"]

    from sentence_transformers import SentenceTransformer
    bert = SentenceTransformer(model_name, cache_folder=os.path.join(root, "sbert"), device=device)

    embedding = bert.encode(text.tolist(), show_progress_bar=True, convert_to_tensor=True)
    embedding = embedding.cpu()

    graph.x = embedding

    return graph


if __name__ == "__main__":
    from tqdm import tqdm

    root = "dataset"
    n_hop = 2

    dataset = get_arxiv_dataset(root, n_hop)


    from models.sentence_embedding import SentenceEmb
    bert = SentenceEmb("multi-qa-distilbert-cos-v1", device="cuda")

    dataloader_var = get_arxiv_dataloader(dataset, "train", batch_size=5, n_way=range(3, 6), n_shot=range(3, 6), n_query=range(10, 24), batch_count=2000, root=root, bert=bert, num_workers=10)
    for i in tqdm(dataloader_var):
        pass

    dataloader = get_arxiv_dataloader(dataset, "train", batch_size=5, n_way=3, n_shot=3, n_query=24, batch_count=2000, root=root, bert=bert, num_workers=10)
    for i in tqdm(dataloader):
        pass
