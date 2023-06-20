'''
    This file provides a simple wrapper for the data loader - it's used to get dataloaders given a dataset name and a few other parameters.
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import random


def sample_with_seed(lst, seed, k=3):
    # from the given list, sample k disjunct subsets with the given seed
    rand = random.Random(seed)
    n = len(lst) // k
    return [rand.sample(lst, n) for _ in range(k)]


def get_dataset_wrap(root, dataset, **kwargs):
    #  rel_sample_rand_seed: If not None, this is the seed used to sample relations for the KG datasets.
    if dataset == "arxiv":
        from data.arxiv import get_arxiv_dataset
        return get_arxiv_dataset(root=os.path.join(root, "arxiv"), **kwargs)
    if dataset == "mag240m":
        from data.mag240m import get_mag240m_dataset
        return get_mag240m_dataset(root=os.path.join(root, "mag240m"), **kwargs)
    if dataset in ["Wiki", "WikiKG90M"]:
        from data.kg import get_kg_dataset
        return get_kg_dataset(root=root, name=dataset, **kwargs)
    elif dataset in ["NELL", "FB15K-237", "ConceptNet"]:
        from data.kg import get_kg_dataset
        return get_kg_dataset(root=root, name=dataset, **kwargs)
    else:
        raise NotImplementedError
