import os
import glob
import json
import torch
import struct
import logging
import copy
import pickle
import numpy as np
import random
import os.path as osp
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch import Tensor
import multiprocessing as mp
import pandas as pd
import math
from tqdm import tqdm
import lmdb
from scipy.sparse import csc_matrix
from sentence_transformers import SentenceTransformer
from pathlib import Path
from experiments.sampler import NeighborSamplerCacheAdj

class Collater:
    def __init__(self):
        pass

    def __call__(self, batch):
        support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, query_subgraphs, negative_triples, negative_subgraphs, curr_rel = list(
            map(list, zip(*batch)))
        if support_subgraphs[0] is None:
            return ((torch.tensor(support_triples), None,
                     torch.tensor(support_negative_triples), None,
                     torch.tensor(query_triples), None,
                     torch.tensor(negative_triples), None),
                    curr_rel)

        support_subgraphs = [item for sublist in support_subgraphs for item in sublist]
        support_negative_subgraphs = [item for sublist in support_negative_subgraphs for item in sublist]
        query_subgraphs = [item for sublist in query_subgraphs for item in sublist]
        negative_subgraphs = [item for sublist in negative_subgraphs for item in sublist]

        return ((support_triples, Batch.from_data_list(support_subgraphs),
                 support_negative_triples, Batch.from_data_list(support_negative_subgraphs),
                 query_triples, Batch.from_data_list(query_subgraphs),
                 negative_triples, Batch.from_data_list(negative_subgraphs)),
                curr_rel)


def get_mid2name_mapping(root_path: str, dataset: str, existing_concepts: set):
    # return mapping from Freebase mid IDs to concept names
    # existing_relations should be a set of existing relations to reduce size of the dict
    # source of mid2name csv : https://github.com/xiaoling/figer/issues/6
    full_path = os.path.join(root_path, dataset, "mid2name.tsv")
    out_dict_path = os.path.join(root_path, dataset, "mid2name_dict.pkl")
    if os.path.exists(out_dict_path):
        return pickle.load(open(out_dict_path, "rb"))
    print("Get Mid2Name Mapping")
    if not os.path.exists(full_path):
        print(
            "WARNING: mid2name.tsv not found. Please put it in the FB15k-237 dataset folder.")
        raise Exception("Download mid2name.tsv from https://github.com/xiaoling/figer/issues/6 and put it in FB15K-237 folder")
    id_to_name = pd.read_csv(full_path, sep="\t", header=None)
    print("read mid2name csv, no. of items in mapping:", len(id_to_name))
    mapping = {}
    for _, row in tqdm(id_to_name.iterrows(), total=len(id_to_name)):
        if row[0] in existing_concepts:
            mapping[row[0]] = row[1]
    pickle.dump(mapping, open(out_dict_path, "wb"))
    return mapping


class PairSubgraphsFewShotDataLoader(DataLoader):
    def __init__(
            self, dataset, batch_size: int = 1,
            shuffle: bool = False,
            **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(),
            **kwargs,
        )

    def next_batch(self):
        return next(iter(self))


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


def ssp_multigraph_to_g(graph, cache=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to torch geometric graph
    """
    if cache and os.path.exists(cache):
        print("Use cache from: ", cache)
        g = torch.load(cache)
        return g, g.edge_attr.max() + 1, g.num_nodes

    edge_list = [[], []]
    edge_features = []
    for i in range(len(graph)):
        edge_list[0].append(graph[i].nonzero()[0])
        edge_list[1].append(graph[i].nonzero()[1])
        edge_features.append(torch.full((len(graph[i].nonzero()[0]),), i))

    edge_list[0] = np.concatenate(edge_list[0])
    edge_list[1] = np.concatenate(edge_list[1])
    edge_index = torch.tensor(np.array(edge_list))

    g = Data(x=None, edge_index=edge_index.long(), edge_attr=torch.cat(edge_features).long(),
             num_nodes=graph[0].shape[0], node_pooling=torch.tensor([[0, 1]]))

    if cache:
        torch.save(g, cache)
        print("Saved graph to", cache)

    return g, len(graph), g.num_nodes


class SubgraphFewshotDataset(Dataset):
    def __init__(self, root, add_traspose_rels=False, shot=1, n_query=3, hop=2, dataset='', mode='dev',
                 kind="union_prune_plus", preprocess=False, preprocess_50neg=False, skip=False, rev=False,
                 use_fix2=False, num_rank_negs=50, inductive=False, orig_test=False, **kwargs):

        self.force_rels = "force_rels" in kwargs and kwargs["force_rels"] is not None and not kwargs["force_rels"] == []
        # force different rels than the ones in the original dataset
        self.root = root
        if orig_test and mode == "test":
            mode = "orig_test"
        self.mode = mode
        self.dataset = dataset
        self.inductive = inductive
        self.rev = rev
        self.postfix = ""
        self.ignore_sampler_cache = self.mode in ["pretrain", "train"]
        if "ignore_sampler_cache" in kwargs:
            self.ignore_sampler_cache = kwargs["ignore_sampler_cache"]
        if "neighbor_sampler" in kwargs:
            self.neighbor_sampler = kwargs["neighbor_sampler"]

        raw_data_paths = os.path.join(root, dataset)

        if "graph_only" in kwargs and kwargs["graph_only"]:
            self.hop = hop
            self.kind = kind
            # we only use this to get the graph for on the fly sampling    
            postfix = "" if not inductive else postfix + "_inductive" 
            path_graph_npy = None
            if dataset == "WikiKG90M":
                print("Loading WikiKG90M train triples")
                from ogb.lsc import WikiKG90Mv2Dataset
                dataset = WikiKG90Mv2Dataset(root=os.path.join(self.root, "ogb-lsc-datasets"))
                path_graph_npy = dataset.train_hrt
            ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation = process_files(raw_data_paths, inductive=inductive, path_graph_npy=path_graph_npy)
            if add_traspose_rels:
                ssp_graph_t = [adj.T for adj in ssp_graph]
                ssp_graph += ssp_graph_t

            self.num_rels_bg = len(relation2id.keys())
            if rev:
                self.num_rels_bg = self.num_rels_bg * 2  # add rev edges
            
            self.ssp_graph = ssp_graph
            self.entity2id = entity2id
            self.relation2id = relation2id
            self.id2entity = id2entity
            self.id2relation = id2relation

            cache = os.path.join(raw_data_paths, f'graph{postfix}.pt')
            if os.path.exists(cache):
                print("Use cache from: ", cache)
                self.graph, _, self.num_nodes_bg = ssp_multigraph_to_g(None, cache)
            else:
                self.graph, _, self.num_nodes_bg = ssp_multigraph_to_g(ssp_graph, cache)
            if self.dataset == "WikiKG90M":
                self.graph.x_id = torch.tensor([int(id2entity[i]) for i in range(self.num_nodes_bg)]).long()
            elif self.dataset == "Wiki":
                self.graph.x_id = torch.arange(self.num_nodes_bg).long()
            return



        self.tasks = json.load(open(os.path.join(raw_data_paths, mode + '_tasks.json')))
        self.tasks_neg = json.load(open(os.path.join(raw_data_paths, mode + '_tasks_neg.json')))

        #  Ds cap: used in the default mode;  default is number of rels; allows different dataset lengths here
        self.dscap = None
        if "dscap" in kwargs:
            self.dscap = kwargs["dscap"]
        print(os.path.join(raw_data_paths, mode + '_tasks.json'))

        if mode == "test" and inductive:
            print("subsample tasks!!!!!!!!!!!!!!!!!!!")
            self.test_tasks_idx = json.load(open(os.path.join(raw_data_paths, 'sample_test_tasks_idx.json')))
            for r in list(self.tasks.keys()):
                if r not in self.test_tasks_idx:
                    self.tasks[r] = []
                else:
                    self.tasks[r] = np.array(self.tasks[r])[self.test_tasks_idx[r]].tolist()

        postfix = "" if not inductive else postfix + "_inductive"
        self.postfix = postfix
        if mode == "pretrain":
            self.tasks = json.load(open(os.path.join(raw_data_paths, mode + f'_tasks{postfix}.json')))
            self.tasks_neg = json.load(open(os.path.join(raw_data_paths, mode + f'_tasks_neg{postfix}.json')))

        self.e1rel_e2 = json.load(open(os.path.join(raw_data_paths, 'e1rel_e2.json')))
        self.all_rels = sorted(list(self.tasks.keys()))
        self.all_rels2id = {self.all_rels[i]: i for i in range(len(self.all_rels))}

        if mode == "test" and inductive:
            for idx, r in enumerate(list(self.all_rels)):
                if len(self.tasks[r]) == 0:
                    del self.tasks[r]
                    print("remove empty tasks!!!!!!!!!!!!!!!!!!!")
            self.all_rels = sorted(list(self.tasks.keys()))

        if "return_all_rels_only" in kwargs and kwargs["return_all_rels_only"]:
            return  # we only use this init code to get all_rels

        self.num_rels = len(self.all_rels)

        if self.force_rels:
            # use the externally provided relations
            self.all_rels = list(kwargs["force_rels"].keys())
            self.tasks = kwargs["force_rels"]
            self.num_rels = len(self.all_rels)
            self.all_rels2id = {self.all_rels[i]: i for i in range(len(self.all_rels))}

        self.few = shot
        self.nq = n_query
        if "shuffle_index" in kwargs and kwargs["shuffle_index"]:
            print("Shuffling index")
            self.shuffle_index = torch.arange(len(self.all_rels)).long()[torch.randperm(len(self.all_rels))]
            print(self.shuffle_index)
        try:
            if mode == "pretrain":
                self.tasks_neg_all = json.load(
                    open(os.path.join(raw_data_paths, mode + f'_tasks_{num_rank_negs}neg{postfix}.json')))
            else:
                self.tasks_neg_all = json.load(
                    open(os.path.join(raw_data_paths, mode + f'_tasks_{num_rank_negs}neg.json')))
            if dataset == "Wiki" or dataset == "WikiKG90M":
                self.tasks_neg_all = json.load(
                    open(os.path.join(raw_data_paths, mode + f'_tasks_{num_rank_negs}neg_subset400.json')))
            self.all_negs = sorted(list(self.tasks_neg_all.keys()))
            self.all_negs2id = {self.all_negs[i]: i for i in range(len(self.all_negs))}
            self.num_all_negs = len(self.all_negs)
        except:
            print(mode + f'_tasks_{num_rank_negs}neg.json', "not exists")

        if mode not in ['train', 'pretrain'] and dataset != "inferwiki_64k":
            self.eval_triples = []
            self.eval_triples_ids = []
            for rel in self.all_rels:
                # sample 5 for each rel
                for i in np.arange(0, len(self.tasks[rel]), 1)[self.few:]:
                    if dataset not in ["Wiki", "WikiKG90M"] or self.tasks[rel][i][0] + self.tasks[rel][i][1] + self.tasks[rel][i][
                        2] in self.all_negs:
                        self.eval_triples.append(self.tasks[rel][i])
                        self.eval_triples_ids.append(i)

            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

        ###### background KG #######
        cache = os.path.join(raw_data_paths, f'graph{postfix}.pt')
        if os.path.exists(cache):
            print("Use cache from: ", cache)
            ssp_graph = None

            with open(os.path.join(raw_data_paths, f'relation2id{postfix}.json'), 'r') as f:
                relation2id = json.load(f)
            with open(os.path.join(raw_data_paths, f'entity2id{postfix}.json'), 'r') as f:
                entity2id = json.load(f)

            id2relation = {v: k for k, v in relation2id.items()}
            id2entity = {v: k for k, v in entity2id.items()}

        else:
            ssp_graph, __, entity2id, relation2id, id2entity, id2relation = process_files(raw_data_paths, inductive=inductive)
            if add_traspose_rels:
                ssp_graph_t = [adj.T for adj in ssp_graph]
                ssp_graph += ssp_graph_t

        self.graph, _, self.num_nodes_bg = ssp_multigraph_to_g(ssp_graph, cache)


        self.num_rels_bg = len(relation2id.keys())
        if rev:
            self.num_rels_bg = self.num_rels_bg * 2  # add rev edges
        #         self.ssp_graph = ssp_graph
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.id2entity = id2entity
        self.id2relation = id2relation

        if dataset == "inferwiki_64k":
            return
        ###### preprocess subgraphs #######

        if rev:
            self.dict_save_path = os.path.join(raw_data_paths, f"preprocessed_rev_fix_new_{kind}_hop={hop}" + postfix)
        else:
            self.dict_save_path = os.path.join(raw_data_paths, f"preprocessed_fix_new_{kind}_hop={hop}" + postfix)
        if use_fix2:
            if rev:
                self.dict_save_path = os.path.join(raw_data_paths,
                                                   f"preprocessed_rev_fix2_new_{kind}_hop={hop}" + postfix)
            else:
                self.dict_save_path = os.path.join(raw_data_paths, f"preprocessed_fix2_new_{kind}_hop={hop}" + postfix)
        print(self.dict_save_path)
        if not os.path.exists(self.dict_save_path):
            os.mkdir(self.dict_save_path)
        new_sampler = True # Use the new sampler for preprocess

        if not self.ignore_sampler_cache:
            postfix_new_sampler = "_new_sampler" if new_sampler else ""
            pos_dict_path = (os.path.join(self.dict_save_path, "pos-%s%s.pt" % (self.mode, postfix_new_sampler)))
            if not os.path.exists(pos_dict_path):
                # self.ignore_sampler_cache = True
                print("**Setting preprocess=True | pos_dict_path does not exist: ", pos_dict_path)
                preprocess = True
            all_neg_dict_path = os.path.join(self.dict_save_path, f"neg_{num_rank_negs}negs-%s%s.pt" % (self.mode, postfix_new_sampler))
            if not os.path.exists(all_neg_dict_path):
                # self.ignore_sampler_cache = True
                print("**Setting preprocess_50neg=True | all_neg_dict_path does not exist: ", all_neg_dict_path)
                preprocess_50neg = True
                if mode == "pretrain":
                    print("Mode is pretrain - don't preprocess 50negs")
                    preprocess_50neg = False

        if self.ignore_sampler_cache or preprocess or preprocess_50neg:
            graph_ns = Data(edge_index=self.graph.edge_index, num_nodes=self.graph.num_nodes)
            self.neighbor_sampler = NeighborSamplerCacheAdj(os.path.join(root, dataset, f"{dataset}_adj.pt"), graph_ns, hop)

        if preprocess or self.force_rels:
            db_path = os.path.join(raw_data_paths, f"subgraphs_fix_new_{kind}_hop=" + str(hop) + postfix)
            if use_fix2:
                db_path = os.path.join(raw_data_paths, f"subgraphs_fix2_new_{kind}_hop=" + str(hop) + postfix)
            if mode == "pretrain":
                db_path = os.path.join(raw_data_paths, f"subgraphs_fix_new_{kind}_hop=" + str(hop) + postfix)
            print(db_path)
            self.main_env = lmdb.open(db_path, readonly=True, max_dbs=4, lock=False)


            self.db_pos = self.main_env.open_db((mode + "_pos").encode())
            self.db_neg = self.main_env.open_db((mode + "_neg").encode())

            self.max_n_label = np.array([3, 3])

            self._preprocess(skip_neg=self.force_rels, new_sampler=new_sampler)

        if preprocess_50neg:
            db_path_50negs = os.path.join(raw_data_paths,
                                          f"subgraphs_fix_new_{kind}_{num_rank_negs}negs_hop=" + str(hop) + postfix)
            if use_fix2:
                db_path_50negs = os.path.join(raw_data_paths,
                                              f"subgraphs_fix2_new_{kind}_{num_rank_negs}negs_hop=" + str(
                                                  hop) + postfix)
            print(db_path_50negs)
            self.main_env = lmdb.open(db_path_50negs, readonly=True, max_dbs=3, lock=False)

            #             self.db_50negs_train = self.main_env.open_db(( "train_neg").encode()) # for fb temp

            self.db_50negs = self.main_env.open_db((mode + "_neg").encode())

            self.max_n_label = np.array([0, 0])
            with self.main_env.begin() as txn:
                self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
                self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self._preprocess_50negs(num_rank_negs, new_sampler=new_sampler)


        postfix_new_sampler = "_new_sampler" if new_sampler else ""
        if (not self.force_rels) and (not preprocess) and (not preprocess_50neg) and (not skip):
            try:
                self.pos_dict = torch.load(os.path.join(self.dict_save_path, "pos-%s%s.pt" % (self.mode, postfix_new_sampler)))
                self.neg_dict = torch.load(os.path.join(self.dict_save_path, "neg-%s%s.pt" % (self.mode, postfix_new_sampler)))
            except:
                print("pos-%s%s.pt" % (self.mode, postfix_new_sampler), "neg-%s%s.pt" % (self.mode, postfix_new_sampler), "not exists")

            try:
                self.all_neg_dict = torch.load(
                    os.path.join(self.dict_save_path, f"neg_{num_rank_negs}negs-%s%s.pt" % (self.mode, postfix_new_sampler)))
            except:
                print(f"neg_{num_rank_negs}negs-%s%s.pt" % (self.mode, postfix_new_sampler), "not exists")


    def __len__(self):
        if self.dscap is not None:
            return self.dscap
        if self.use_50negs_mode:
            return len(self.eval_triples)
        return self.num_rels if self.num_rels != 0 else 1  # dummy train

    def _save_torch_geometric(self, index, skip_neg=False, new_sampler=False):
        curr_rel = self.all_rels[index]
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        if not skip_neg:
            curr_tasks_neg = self.tasks_neg[curr_rel]
            curr_tasks_neg_idx = np.arange(0, len(curr_tasks_neg), 1)

        pos_edge_index, pos_x, pos_x_id, pos_edge_attr, pos_n_size, pos_e_size = [], [], [], [], [], []
        neg_edge_index, neg_x, neg_x_id, neg_edge_attr, neg_n_size, neg_e_size = [], [], [], [], [], []


        with self.main_env.begin(db=self.db_pos) as txn:
            for idx, i in enumerate(curr_tasks_idx):
                #               print(curr_rel, i, curr_tasks[i])
                str_id = curr_rel.encode() + '{:08}'.format(i).encode('ascii')
                nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
                if not new_sampler:
                    d = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
                else:
                    d = self.get_new_subgraph(curr_tasks[i], curr_rel)
                if nodes_pos[0] == nodes_pos[1]:
                    print(curr_rel, index, i, curr_tasks[i])
                pos_edge_index.append(d.edge_index)
                pos_x.append(d.x)
                pos_x_id.append(d.x_id)
                pos_edge_attr.append(d.edge_attr)
                pos_n_size.append(d.x.shape[0])
                pos_e_size.append(d.edge_index.shape[1])
        if not skip_neg:
            with self.main_env.begin(db=self.db_neg) as txn:
                for idx, i in enumerate(curr_tasks_neg_idx):
                    str_id = curr_rel.encode() + '{:08}'.format(i).encode('ascii')
                    nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                    if not new_sampler:
                        d = self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg)
                    else:
                        d = self.get_new_subgraph(curr_tasks_neg[i], curr_rel) 
                    if nodes_neg[0] == nodes_neg[1]:
                        print("neg", curr_rel, index, i, curr_tasks[i])
                    neg_edge_index.append(d.edge_index)
                    neg_x.append(d.x)
                    neg_x_id.append(d.x_id)
                    neg_edge_attr.append(d.edge_attr)
                    neg_n_size.append(d.x.shape[0])
                    neg_e_size.append(d.edge_index.shape[1])
            return torch.cat(pos_edge_index, 1), torch.cat(pos_x, 0), torch.cat(pos_x_id, 0), torch.cat(pos_edge_attr, 0), \
                   torch.LongTensor(pos_n_size), torch.LongTensor(pos_e_size), torch.cat(neg_edge_index, 1), \
                   torch.cat(neg_x, 0), torch.cat(neg_x_id, 0), torch.cat(neg_edge_attr, 0), torch.LongTensor(neg_n_size), \
                   torch.LongTensor(neg_e_size)
        else:
            return torch.cat(pos_edge_index, 1), torch.cat(pos_x, 0), torch.cat(pos_x_id, 0), torch.cat(pos_edge_attr, 0), \
                   torch.LongTensor(pos_n_size), torch.LongTensor(pos_e_size), [], [], [], [], [], []

    def dict_to_torch_geometric(self, index, data_dict):

        if index == 0:
            task_index = 0
            start_e = 0
            start_n = 0
        else:
            task_index = data_dict["task_offsets"][index - 1]
            start_e = data_dict['e_size'][task_index - 1]
            start_n = data_dict['n_size'][task_index - 1]

        task_index_end = data_dict["task_offsets"][index]

        graphs = []
        for i in range(task_index_end - task_index):
            end_e = data_dict['e_size'][task_index + i]
            end_n = data_dict['n_size'][task_index + i]
            edge_index = data_dict['edge_index'][:, start_e:end_e]
            x = data_dict['x'][start_n:end_n]
            x_id = data_dict['x_id'][start_n:end_n]
            edge_attr = data_dict['edge_attr'][start_e:end_e]
            # Reshape edge_attr
            edge_attr = edge_attr.view(-1, 1)
            graphs.append(Data(edge_index=edge_index, x=x, x_id=x_id, edge_attr=edge_attr,
                               node_pooling=torch.tensor([[0, 1]])))
            start_e = end_e
            start_n = end_n

        return graphs

    def _preprocess_50negs(self, num_rank_negs, new_sampler=False):
        print("start preprocessing 50negs for %s" % self.mode)
        all_neg_edge_index, all_neg_x, all_neg_x_id, all_neg_edge_attr, all_neg_n_size, all_neg_e_size = [], [], [], [], [], []
        task_offsets_neg = []
        for index in tqdm(range(self.num_all_negs)):
            curr_rel = self.all_negs[index]
            curr_tasks_neg = self.tasks_neg_all[curr_rel]
            curr_tasks_neg_idx = np.arange(0, len(curr_tasks_neg), 1)
            neg_edge_index, neg_x, neg_x_id, neg_edge_attr, neg_n_size, neg_e_size = [], [], [], [], [], []
            #             with mp.Pool() as p:
            #                 for d, idx in tqdm(p.imap(self._prepare_subgraphs_helper, list(range(len(curr_tasks_neg_idx)))), total=len(curr_tasks_neg_idx), leave = False):
            with self.main_env.begin(db=self.db_50negs) as txn:
                for idx, i in enumerate(curr_tasks_neg_idx):
                    str_id = curr_rel.encode() + '{:08}'.format(i).encode('ascii')
                    #                     if txn.get(str_id) is not None:
                    #                         print("exists")
                    nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                    if not new_sampler:
                        d = self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg)
                    else:
                        d = self.get_new_subgraph(curr_tasks_neg[i], curr_rel)
                    neg_edge_index.append(d.edge_index)
                    neg_x.append(d.x)
                    neg_x_id.append(d.x_id)
                    neg_edge_attr.append(d.edge_attr)
                    neg_n_size.append(d.x.shape[0])
                    neg_e_size.append(d.edge_index.shape[1])

            all_neg_edge_index.append(torch.cat(neg_edge_index, 1))
            all_neg_x.append(torch.cat(neg_x, 0))
            all_neg_x_id.append(torch.cat(neg_x_id, 0))
            all_neg_edge_attr.append(torch.cat(neg_edge_attr, 0))
            all_neg_n_size.append(torch.LongTensor(neg_n_size))
            all_neg_e_size.append(torch.LongTensor(neg_e_size))
            task_offsets_neg.append(len(torch.LongTensor(neg_n_size)))

        print("concat all")

        all_neg_edge_index = torch.cat(all_neg_edge_index, 1)
        all_neg_x = torch.cat(all_neg_x, 0)
        all_neg_x_id = torch.cat(all_neg_x_id, 0)
        all_neg_edge_attr = torch.cat(all_neg_edge_attr, 0)

        all_neg_n_size = torch.cat(all_neg_n_size)
        all_neg_e_size = torch.cat(all_neg_e_size)

        all_neg_n_size = torch.cumsum(all_neg_n_size, 0)
        all_neg_e_size = torch.cumsum(all_neg_e_size, 0)

        task_offsets_neg = torch.tensor(task_offsets_neg)
        task_offsets_neg = torch.cumsum(task_offsets_neg, 0)

        save_path = self.dict_save_path

        neg_save_dict = {
            'edge_index': all_neg_edge_index,
            'x': all_neg_x,
            'x_id': all_neg_x_id,
            'edge_attr': all_neg_edge_attr,
            'task_offsets': task_offsets_neg,
            'n_size': all_neg_n_size,
            'e_size': all_neg_e_size
        }
        postfix = "_new_sampler" if new_sampler else ""
        print("saving to", os.path.join(save_path, f"neg_{num_rank_negs}negs-%s%s.pt" % (self.mode, postfix)))
        torch.save(neg_save_dict, os.path.join(save_path, f"neg_{num_rank_negs}negs-%s%s.pt" % (self.mode, postfix)))
        self.all_neg_dict = neg_save_dict

    def _preprocess(self, skip_neg=False, new_sampler=False):
        print("start preprocessing %s" % self.mode)
        all_pos_edge_index, all_pos_x, all_pos_x_id, all_pos_edge_attr, all_pos_n_size, all_pos_e_size = [], [], [], [], [], []
        all_neg_edge_index, all_neg_x, all_neg_x_id, all_neg_edge_attr, all_neg_n_size, all_neg_e_size = [], [], [], [], [], []
        task_offsets_pos = []
        task_offsets_neg = []
        for index in tqdm(range(self.num_rels)):
            pos_edge_index, pos_x, pos_x_id, pos_edge_attr, pos_n_size, pos_e_size, neg_edge_index, neg_x, neg_x_id, neg_edge_attr, neg_n_size, neg_e_size = self._save_torch_geometric(
                index, skip_neg=skip_neg, new_sampler=new_sampler)
            all_pos_edge_index.append(pos_edge_index)
            all_pos_x.append(pos_x)
            all_pos_x_id.append(pos_x_id)
            all_pos_edge_attr.append(pos_edge_attr)
            all_pos_n_size.append(pos_n_size)
            all_pos_e_size.append(pos_e_size)
            task_offsets_pos.append(len(pos_n_size))

            all_neg_edge_index.append(neg_edge_index)
            all_neg_x.append(neg_x)
            all_neg_x_id.append(neg_x_id)
            all_neg_edge_attr.append(neg_edge_attr)
            all_neg_n_size.append(neg_n_size)
            all_neg_e_size.append(neg_e_size)
            task_offsets_neg.append(len(neg_n_size))

        print("concat all")
        all_pos_edge_index = torch.cat(all_pos_edge_index, 1)
        all_pos_x = torch.cat(all_pos_x, 0)
        all_pos_x_id = torch.cat(all_pos_x_id, 0)
        all_pos_edge_attr = torch.cat(all_pos_edge_attr, 0)

        all_neg_edge_index = torch.cat(all_neg_edge_index, 1)
        all_neg_x = torch.cat(all_neg_x, 0)
        all_neg_x_id = torch.cat(all_neg_x_id, 0)
        all_neg_edge_attr = torch.cat(all_neg_edge_attr, 0)

        all_pos_n_size = torch.cat(all_pos_n_size)
        all_pos_e_size = torch.cat(all_pos_e_size)
        all_neg_n_size = torch.cat(all_neg_n_size)
        all_neg_e_size = torch.cat(all_neg_e_size)

        all_pos_n_size = torch.cumsum(all_pos_n_size, 0)
        all_pos_e_size = torch.cumsum(all_pos_e_size, 0)
        all_neg_n_size = torch.cumsum(all_neg_n_size, 0)
        all_neg_e_size = torch.cumsum(all_neg_e_size, 0)

        task_offsets_pos = torch.tensor(task_offsets_pos)
        task_offsets_pos = torch.cumsum(task_offsets_pos, 0)
        task_offsets_neg = torch.tensor(task_offsets_neg)
        task_offsets_neg = torch.cumsum(task_offsets_neg, 0)

        save_path = self.dict_save_path
        pos_save_dict = {
            'edge_index': all_pos_edge_index,
            'x': all_pos_x,
            'x_id': all_pos_x_id,
            'edge_attr': all_pos_edge_attr,
            'task_offsets': task_offsets_pos,
            'n_size': all_pos_n_size,
            'e_size': all_pos_e_size
        }

        neg_save_dict = {
            'edge_index': all_neg_edge_index,
            'x': all_neg_x,
            'x_id': all_neg_x_id,
            'edge_attr': all_neg_edge_attr,
            'task_offsets': task_offsets_neg,
            'n_size': all_neg_n_size,
            'e_size': all_neg_e_size
        }

        print("saving")
        postfix = "_new_sampler" if new_sampler else ""
        if not self.force_rels:
            torch.save(pos_save_dict, os.path.join(save_path, "pos-%s%s.pt" % (self.mode, postfix)))
            torch.save(neg_save_dict, os.path.join(save_path, "neg-%s%s.pt" % (self.mode, postfix)))
        else:
            print("Not saving - using different rels")
        self.pos_dict = pos_save_dict
        self.neg_dict = neg_save_dict

    def get_length_multiclass(self, max_n_class=3):
        '''
        In multiclass case, length will typically be smaller.
        :param max_n_class:
        :return:
        '''
        if self.batchsz is not None:
            return self.batchsz
        return math.ceil(self.num_rels / max_n_class)

    def _gen_batch_multiclass(self, batchsz=None, max_n_class=3):
        '''
        Generate batch info similar to G-Meta
        ("batch size" is basically how many times we sample max_n_classes from all classes and return this as a separate task)
        :param batchsz: If None, it will be set automatically with math.ceil(self.num_rels / max_n_class)
        :return:
        '''

        if batchsz is None:
            batchsz = math.ceil(30 * self.num_rels / max_n_class)
            print("WARNING: you didn't set dataset size - auto setting to", batchsz)
        self.batchsz = batchsz
        batches = []
        for i in range(batchsz):
            batches.append(np.random.choice(self.num_rels, size=max_n_class, replace=False))
        self.batches = batches

    def get_task_multiclass(self, index, max_n_class=3):
        '''
        Similar to __getitem__, except returns only positive subgraphs for max_n_class classes.
        :param max_n_class: Maximum number of classes (relations) that should be returned
        :return:
        '''
        assert index < self.get_length_multiclass(max_n_class=max_n_class)

        rels_idx = np.arange(0, self.num_rels, 1)
        sampled_classes = self.batches[index]

        curr_rels = [self.all_rels[sampled_class] for sampled_class in sampled_classes]
        curr_tasks_list = [self.tasks[curr_rel] for curr_rel in curr_rels]
        result = []

        for i, curr_tasks in enumerate(curr_tasks_list):
            curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
            if self.nq is not None:
                curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few + self.nq, replace=False)
            support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
            query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]
            curr_rel = curr_rels[i]

            all_pos_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.pos_dict)

            support_subgraphs = []
            query_subgraphs = []

            for idx, i in enumerate(curr_tasks_idx):
                if self.mode == "test" and self.inductive:
                    subgraph_pos = all_pos_graphs[self.test_tasks_idx[curr_rel][i]]
                else:
                    subgraph_pos = all_pos_graphs[i]
                #             subgraph_pos = all_pos_graphs[i]
                if idx < self.few:
                    support_subgraphs.append(subgraph_pos)
                else:
                    query_subgraphs.append(subgraph_pos)
            result.append((support_triples, support_subgraphs, query_triples, query_subgraphs, curr_rel))

        return result, curr_rels
    
    def get_new_subgraph(self, reiplet, r_label): 
        e = [self.entity2id[reiplet[0]], self.entity2id[reiplet[2]]]
        nodes, edge_index, edge_id = self.neighbor_sampler.sample_node(e)

        subgraph = get_subgraph(self.graph, torch.tensor(nodes))

        # remove the (0,1) target edge 
        if r_label in self.relation2id:
            index = (torch.tensor([0, 1]) == subgraph.edge_index.transpose(0, 1)).all(1)
            index = index & (subgraph.edge_attr == self.relation2id[r_label])
            if index.any():
                subgraph.edge_index = subgraph.edge_index.transpose(0, 1)[~index].transpose(0, 1)
                subgraph.edge_attr = subgraph.edge_attr[~index]

        
        # add reverse edges 
        if self.rev:
            subgraph.edge_index = torch.cat([subgraph.edge_index, subgraph.edge_index.flip(0)], 1)
            subgraph.edge_attr = torch.cat([subgraph.edge_attr, self.num_rels_bg - subgraph.edge_attr], 0)

        # One hot encode the node label feature and concat to n_features
        n_nodes = subgraph.num_nodes

        subgraph.x = torch.LongTensor(nodes)
        subgraph.x_id = torch.LongTensor(nodes)

        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr
        row = edge_index[0]
        col = edge_index[1]
        idx = col.new_zeros(col.numel() + 1)
        idx[1:] = row
        idx[1:] *= subgraph.x.shape[0]
        idx[1:] += col
        perm = idx[1:].argsort()
        row = row[perm]
        col = col[perm]
        edge_attr = edge_attr[perm]
        edge_index = torch.stack([row, col], 0)

        subgraph.edge_index = edge_index
        subgraph.edge_attr = edge_attr

        return subgraph

    def __getitem__(self, index):
        # get current relation and current candidates
        if hasattr(self, "shuffle_index") and self.shuffle_index is not None:
            index = self.shuffle_index[index]
        if index >= self.num_rels:
            index = index % self.num_rels  # correct the index...
        curr_rel = self.all_rels[index]

        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        if self.nq is not None:
            curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few + self.nq, replace=False)

        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]
        if not self.ignore_sampler_cache:
            all_pos_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.pos_dict)
            all_neg_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.neg_dict)

        ### extract subgraphs     
        support_subgraphs = []
        query_subgraphs = []
        for idx, i in enumerate(curr_tasks_idx):

            if self.mode == "test" and self.inductive:
                e = curr_tasks[self.test_tasks_idx[curr_rel][i]]
                if self.ignore_sampler_cache:
                    subgraph_pos = self.get_new_subgraph(e, curr_rel)
                else:
                    subgraph_pos = all_pos_graphs[self.test_tasks_idx[curr_rel][i]]

                
            else:
                e = curr_tasks[i]
                if self.ignore_sampler_cache:
                    subgraph_pos = self.get_new_subgraph(e, curr_rel)
                else:
                    subgraph_pos = all_pos_graphs[i]
            if idx < self.few:
                support_subgraphs.append(subgraph_pos)
            else:
                query_subgraphs.append(subgraph_pos)

        curr_tasks_neg = self.tasks_neg[curr_rel]
        curr_tasks_neg_idx = curr_tasks_idx
        if self.dataset == "inferwiki_64k":
            ### inferwiki does not have paired sampling for negs
            curr_tasks_neg_idx = np.arange(0, len(curr_tasks_neg), 1)
            if self.nq is not None:
                curr_tasks_neg_idx = np.random.choice(curr_tasks_neg_idx, self.few + self.nq, replace=False)

        support_negative_triples = [curr_tasks_neg[i] for i in curr_tasks_neg_idx[:self.few]]
        negative_triples = [curr_tasks_neg[i] for i in curr_tasks_neg_idx[self.few:]]


        support_negative_subgraphs = []
        negative_subgraphs = []
        for idx, i in enumerate(curr_tasks_neg_idx):


            if self.mode == "test" and self.inductive:
                e = curr_tasks_neg[self.test_tasks_idx[curr_rel][i]]
                if self.ignore_sampler_cache:
                    subgraph_neg = self.get_new_subgraph(e, curr_rel)
                else:
                    subgraph_neg = all_neg_graphs[self.test_tasks_idx[curr_rel][i]]
            else:
                e = curr_tasks_neg[i]
                if self.ignore_sampler_cache:
                    subgraph_neg = self.get_new_subgraph(e, curr_rel)
                else:
                    subgraph_neg = all_neg_graphs[i]

            if (self.mode in ["train", "pretrain"] and self.dataset in ['NELL',
                                                                        'FB15K-237'] and not self.inductive):
                # choose 1 neg from 50
                e1, r, e2 = curr_tasks[i]
                e = random.choice(self.tasks_neg_all[e1+r+e2])
                if self.ignore_sampler_cache:
                    subgraph_neg = self.get_new_subgraph(e, curr_rel)
                else:
                    e1, r, e2 = curr_tasks[i]
                    all_50_neg_graphs = self.dict_to_torch_geometric(self.all_negs2id[e1 + r + e2], self.all_neg_dict)
                    subgraph_neg = random.choice(all_50_neg_graphs)

            if idx < self.few:
                support_negative_subgraphs.append(subgraph_neg)
            else:
                negative_subgraphs.append(subgraph_neg)
                
        return support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, query_subgraphs, negative_triples, negative_subgraphs, curr_rel

    def next_one_on_eval(self, index):
        # get current triple
        query_triple = self.eval_triples[index]
        curr_rel = query_triple[1]
        curr_rel_neg = query_triple[0] + query_triple[1] + query_triple[2]
        curr_task = self.tasks[curr_rel]
        if not self.ignore_sampler_cache:
            all_pos_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.pos_dict)
            all_neg_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.neg_dict)
            all_50_neg_graphs = self.dict_to_torch_geometric(self.all_negs2id[curr_rel_neg], self.all_neg_dict)

        # get support triples
        support_triples_idx = np.arange(0, len(curr_task), 1)[:self.few]
        support_triples = []
        support_subgraphs = []
        for idx, i in enumerate(support_triples_idx):
            support_triples.append(curr_task[i])
            if self.mode == "test" and self.inductive:
                e = curr_task[self.test_tasks_idx[curr_rel][i]]
                if self.ignore_sampler_cache:
                    subgraph_pos = self.get_new_subgraph(e, curr_rel)
                else:
                    subgraph_pos = all_pos_graphs[self.test_tasks_idx[curr_rel][i]]
            else:
                e = curr_task[i]
                if self.ignore_sampler_cache:
                    subgraph_pos = self.get_new_subgraph(e, curr_rel)
                else:
                    subgraph_pos = all_pos_graphs[i]
            support_subgraphs.append(subgraph_pos)

        query_triples = [query_triple]
        query_subgraphs = []


        if self.mode == "test" and self.inductive:
            e = curr_task[self.test_tasks_idx[curr_rel][self.eval_triples_ids[index]]]
            if self.ignore_sampler_cache:
                subgraph_pos = self.get_new_subgraph(e, curr_rel)
            else:
                subgraph_pos = all_pos_graphs[self.test_tasks_idx[curr_rel][self.eval_triples_ids[index]]]
        else:
            e = curr_task[self.eval_triples_ids[index]]
            if self.ignore_sampler_cache:
                subgraph_pos = self.get_new_subgraph(e, curr_rel)
            else:
                subgraph_pos = all_pos_graphs[self.eval_triples_ids[index]]

        query_subgraphs.append(subgraph_pos)

        # construct support negative

        curr_task_neg = self.tasks_neg[curr_rel]
        support_negative_triples_idx = support_triples_idx
        support_negative_triples = []
        support_negative_subgraphs = []
        for idx, i in enumerate(support_negative_triples_idx):
            support_negative_triples.append(curr_task_neg[i])

            if self.mode == "test" and self.inductive:
                e = curr_task_neg[self.eval_triples_ids[index]]
                if self.ignore_sampler_cache:
                    subgraph_neg = self.get_new_subgraph(e, curr_rel)
                else:
                    subgraph_neg = all_neg_graphs[self.test_tasks_idx[curr_rel][i]]
            else:
                e = curr_task_neg[self.eval_triples_ids[index]]
                if self.ignore_sampler_cache:
                    subgraph_neg = self.get_new_subgraph(e, curr_rel)
                else:
                    subgraph_neg = all_neg_graphs[i]

            support_negative_subgraphs.append(subgraph_neg)

        ### 50 query negs
        curr_task_50neg = self.tasks_neg_all[curr_rel_neg]
        negative_triples_idx = np.arange(0, len(curr_task_50neg), 1)
        negative_triples = []
        negative_subgraphs = []
        for idx, i in enumerate(negative_triples_idx):
            negative_triples.append(curr_task_50neg[i])
            e = curr_task_50neg[i]
            if self.ignore_sampler_cache:
                negative_subgraphs.append(self.get_new_subgraph(e, curr_rel))
            else:
                negative_subgraphs.append(all_50_neg_graphs[i])


        return support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, query_subgraphs, negative_triples, negative_subgraphs, curr_rel

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        if nodes[0] == nodes[1]:
            print(nodes)
            print("self-loop...")
            nodes = nodes[:2]
            subgraph = Data(edge_index=torch.zeros([2, 0]), edge_attr=torch.zeros([0]), num_nodes=2, node_pooling=torch.tensor([[0, 1]]))
        else:
            subgraph = get_subgraph(self.graph, torch.tensor(nodes))
        index = (torch.tensor([0, 1]) == subgraph.edge_index.transpose(0, 1)).all(1)
        index = index & (subgraph.edge_attr == r_label)
        if index.any():
            subgraph.edge_index = subgraph.edge_index.transpose(0, 1)[~index].transpose(0, 1)
            subgraph.edge_attr = subgraph.edge_attr[~index]

        # add reverse edges 
        if self.rev:
            subgraph.edge_index = torch.cat([subgraph.edge_index, subgraph.edge_index.flip(0)], 1)
            subgraph.edge_attr = torch.cat([subgraph.edge_attr, self.num_rels_bg - subgraph.edge_attr], 0)

        # One hot encode the node label feature and concat to n_features
        n_nodes = subgraph.num_nodes
        n_labels = n_labels.astype(int)

        label_feats = np.zeros((n_nodes, 6))
        label_feats[0] = [1, 0, 0, 0, 1, 0]
        label_feats[1] = [0, 1, 0, 1, 0, 0]

        subgraph.x = torch.FloatTensor(label_feats)
        subgraph.x_id = torch.LongTensor(nodes)


        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr
        row = edge_index[0]
        col = edge_index[1]
        idx = col.new_zeros(col.numel() + 1)
        idx[1:] = row
        idx[1:] *= subgraph.x.shape[0]
        idx[1:] += col
        perm = idx[1:].argsort()
        row = row[perm]
        col = col[perm]
        edge_attr = edge_attr[perm]
        edge_index = torch.stack([row, col], 0)

        subgraph.edge_index = edge_index
        subgraph.edge_attr = edge_attr

        return subgraph


def process_files(data_path, use_cache=True, inductive=False, path_graph_npy=None):
    entity2id = {}
    relation2id = {}

    postfix = "" if not inductive else postfix + "_inductive"
    relation2id_path = os.path.join(data_path, f'relation2id{postfix}.json')

    is_preprocessed = os.path.exists(relation2id_path) and use_cache and not "NELL" in data_path and not "FB15K-237" in data_path and not "ConceptNet" in data_path
    
    if use_cache and os.path.exists(relation2id_path):
        print("Use cache from: ", relation2id_path)
        with open(relation2id_path, 'r') as f:
            relation2id = json.load(f)

    entity2id_path = os.path.join(data_path, f'entity2id{postfix}.json')
    if use_cache and os.path.exists(entity2id_path):
        print("Use cache from: ", entity2id_path)
        with open(entity2id_path, 'r') as f:
            entity2id = json.load(f)
    adj_list = []
    triplets = {}

    ent = 0
    rel = 0
    if not is_preprocessed:
        for mode in ['bg']:  # assuming only one kind of background graph for now
            data = []
            if path_graph_npy is None:
                file_path = os.path.join(data_path, f'path_graph{postfix}.json')
                with open(file_path) as f:
                    file_data = json.load(f)
            else:
                file_data = path_graph_npy

            for triplet in tqdm(file_data):
                if triplet[0] not in entity2id:
                    entity2id[triplet[0]] = ent
                    ent += 1
                if triplet[2] not in entity2id:
                    entity2id[triplet[2]] = ent
                    ent += 1
                if triplet[1] not in relation2id:
                    relation2id[triplet[1]] = rel
                    rel += 1

                # Save the triplets corresponding to only the known relations
                if triplet[1] in relation2id:
                    data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

            triplets[mode] = np.array(data)
       
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation. Note that this is constructed only from the train data.
    if not is_preprocessed:
        for i in tqdm(range(len(relation2id))):
            idx = np.argwhere(triplets['bg'][:, 2] == i).flatten()
            adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                        (triplets['bg'][:, 0][idx], triplets['bg'][:, 1][idx])),
                                       shape=(len(entity2id), len(entity2id))))

    def intstr(item):
        try:
            return int(item)
        except:
            return str(item)
    if not os.path.exists(relation2id_path):
        print("Writing rel2id")
        with open(relation2id_path, 'w') as f:
            relation2id = {intstr(key): value for key, value in relation2id.items()}
            json.dump(relation2id, f)

    if not os.path.exists(entity2id_path):
        print("Writing entity2id")
        with open(entity2id_path, 'w') as f:
            entity2id = {intstr(key): value for key, value in entity2id.items()}
            json.dump(entity2id, f)
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def index_to_mask(index, size=None):
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def get_subgraph(graph, nodes):
    """ from torch_geomtric"""
    """
        get induced subgraph of the given nodes
        -----------------------------------------
        nodes: list
        graph: PyG or similar Data object
        
    """
    #     print(nodes)
    relabel_nodes = True
    # nodes = torch.unique(nodes)

    device = graph.edge_index.device

    num_nodes = graph.num_nodes
    subset = index_to_mask(nodes, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[graph.edge_index[0]] & node_mask[graph.edge_index[1]]
    edge_index = graph.edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=device)
        node_idx[nodes] = torch.arange(subset.sum().item(), device=device)
        edge_index = node_idx[edge_index]
    num_nodes = nodes.size(0)
    data = copy.copy(graph)
    for key, value in data:
        if key == 'edge_index':
            data.edge_index = edge_index
        elif key == 'num_nodes':
            data.num_nodes = num_nodes
        elif isinstance(value, Tensor):
            if graph.is_node_attr(key):
                data[key] = value[subset]
            elif graph.is_edge_attr(key):
                data[key] = value[edge_mask]
    return data


class SubgraphFewshotDatasetRankTail(SubgraphFewshotDataset):
    def __len__(self):
        return len(self.eval_triples)

    def __getitem__(self, index):
        return self.next_one_on_eval(index)


class SubgraphFewshotDatasetWithTextFeats(SubgraphFewshotDataset):
    def __init__(self, *args, **kwargs):
        super(SubgraphFewshotDatasetWithTextFeats, self).__init__(*args, **kwargs)
        if self.dataset not in ["FB15K-237", "NELL", "NELL_newsplit", "ConceptNet", "Wiki", "WikiKG90M"]:
            raise NotImplementedError  # only NELL, FB15k-237 and ConceptNet have text features for now
        if self.dataset == "FB15K-237":
            self.mid2name = get_mid2name_mapping(root_path=self.root, dataset=self.dataset,
                                                 existing_concepts=set(list(self.entity2id.keys())))
        else:
            self.mid2name = None
        self.bert = kwargs["bert"] # default "multi-qa-distilbert-cos-v1"
        self.root = kwargs["root"]
        self.dataset = kwargs["dataset"]
        self.device = kwargs["device"]
        self.mode = kwargs["mode"]  # train / test / validation
        self.pretrained_embeddings = None
        if self.dataset == "WikiKG90M":
            from ogb.lsc import WikiKG90Mv2Dataset
            dataset = WikiKG90Mv2Dataset(root=os.path.join(self.root, "ogb-lsc-datasets"))
            # these features are loaded into memory on-the-fly!
            self.disk_features = {"node": dataset.entity_feat, "rel": dataset.relation_feat}
            return
        self.disk_features = None

        self.text_feats = self._preprocess_text_feats(self.bert)
        if self.dataset not in ["Wiki"] and not ("graph_only" in kwargs and kwargs["graph_only"]):
            self.text_feats.update(self._preprocess_text_feats_mode_specific(self.bert))

        if "embeddings_model" in kwargs and kwargs["embeddings_model"]:
            if kwargs["embeddings_model"] == "random":
                # random node and edge embeddings
                self.pretrained_embeddings = {
                    "node": torch.randn((len(self.entity2id), 100)),
                    "rel": torch.randn((len(self.relation2id), 100))
                }
                print("Use random node/edge embeddings")
            else:
                raise Exception("To use KG Embedding, please properly setup load_embed function and embedding path as in https://github.com/snap-stanford/csr/blob/main/models.py and https://github.com/snap-stanford/csr/blob/main/README.md")
            
                use_ours = self.dataset != "NELL"
                embeddings_rel = load_embed(os.path.join(self.root, self.dataset), self.dataset, embed_model=kwargs["embeddings_model"], inductive=False, use_ours=use_ours)
                embeddings_node = load_embed(os.path.join(self.root, self.dataset), self.dataset,
                                             embed_model=kwargs["embeddings_model"],
                                             inductive=False, use_ours=use_ours,
                                             load_ent=True)
                print("Loaded pretrained embeddings")
                self.pretrained_embeddings = {'rel': torch.from_numpy(embeddings_rel).float(),
                                              'node': torch.from_numpy(embeddings_node).float()}
                print("Loaded pretrained", kwargs["embeddings_model"], "embeddings")

    def _add_text_feats_to_pyg_base(self, data: Data):
        #  Adds text features to a PyG data object.
        #  Here we assume that the 0th node is the head and the 1st node is the tail.
        x_text = [self.id2entity.get(i.item(), str(i.item())) for i in data.x_id]
        edge_attr_text = [self.id2relation.get(i.item(), str(i.item())) for i in data.edge_attr]
        if self.disk_features is not None:
            # load Wiki kg features from disk
            data.x = self.disk_features["node"][data.x_id]
            data.edge_attr = self.disk_features["rel"][data.edge_attr.flatten().long()]
        elif self.pretrained_embeddings is not None:
            # use the provided KG embedddings of relations and entities
            data.x = self.pretrained_embeddings["node"][data.x_id]
            data.edge_attr = self.pretrained_embeddings["rel"][data.edge_attr.flatten().long()]
        else:
            if self.mid2name is not None:
                x_text = [self.mid2name.get(i, i) for i in x_text]
            data.x = torch.stack([self.text_feats[i] for i in x_text], dim=0)
            if len(edge_attr_text) is not 0:
                data.edge_attr = torch.stack([self.text_feats[i] for i in edge_attr_text], dim=0)
            else:
                data.edge_attr = torch.empty((0, 768)).float()
                data.edge_index = data.edge_index.long()
        return data


    def _add_text_feats_to_pyg(self, data: Data):
        data = self._add_text_feats_to_pyg_base(data)
        data.x = torch.cat([data.x, torch.zeros(data.x.shape[0], 2)], dim=1)
        data.x[0, -1] = 1.
        data.x[1, -2] = 1.  # always flag the head and tail nodes
        return data

    def _preprocess_text_feats(self, model_name):
        print("MODEL NAME: ", model_name)
        cache_path = os.path.join(self.root, self.dataset, "preproc_text_feats")
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        cache_filename = os.path.join(cache_path, "text_feats_{}_{}.pkl".format(model_name.replace("/", "_"), self.postfix))
        # additional file for subset-specific things
        if self.dataset == "Wiki":
            text_dict_path = os.path.join(self.root, self.dataset, "text_features_web_scraped.pb")
            self.text_dict = pickle.load(open(text_dict_path, "rb"))
        if os.path.exists(cache_filename):
            print("Loading text features from ", cache_filename)
            texts, embeddings = torch.load(cache_filename)
            text_to_emb = {text: emb for text, emb in zip(texts, embeddings)}
            return text_to_emb
        print("Preprocessing text features for {}....".format(self.dataset))
        bert = SentenceTransformer(model_name, cache_folder=os.path.join(self.root, "sbert"), device=self.device)
        entity_list = list(self.id2entity.values())
        if self.dataset != "Wiki":
            entity_list += ["Head: " + i for i in entity_list] + ["Tail: " + i for i in entity_list]
            # Also add head and tail encodings...
        all_text = set(entity_list + list(self.id2relation.values()))
        if self.mid2name is not None:
            #  also add some text features for FreeBase entities
            additional_names = [self.mid2name[i] for i in all_text if i in self.mid2name]
            additional_names += ["Head: " + i for i in additional_names] + ["Tail: " + i for i in additional_names]
            all_text = all_text.union(set(additional_names))
        all_text = list(all_text)
        if self.dataset == "Wiki":
            print("Using the fetched text features for Wiki dataset")
            all_text = [self.text_dict.get(i, i) for i in all_text]
        embeddings = bert.encode(all_text, show_progress_bar=True, convert_to_tensor=True,
                                normalize_embeddings=True, batch_size=1024)
        embeddings = embeddings.cpu()
        torch.save((all_text, embeddings), cache_filename)
        print("Saved to ", cache_filename)

        return {text: emb for text, emb in zip(all_text, embeddings)}
    def _preprocess_text_feats_mode_specific(self, model_name):
        cache_path = os.path.join(self.root, self.dataset, "preproc_text_feats")
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        cache_filename = os.path.join(cache_path, "text_feats_{}_{}_mode_{}.pkl".format(model_name.replace("/", "_"), self.postfix, self.mode))
        # additional file for subset-specific things
        if os.path.exists(cache_filename):
            print("Loading text features from ", cache_filename)
            texts, embeddings = torch.load(cache_filename)
            text_to_emb = {text: emb for text, emb in zip(texts, embeddings)}
            return text_to_emb
        print("Preprocessing text features for {}...".format(self.dataset))
        bert = SentenceTransformer(model_name, cache_folder=os.path.join(self.root, "sbert"), device=self.device)
        all_text = list(self.tasks.keys())
        if self.dataset == "Wiki":
            all_text = [self.text_dict.get(i, i) for i in all_text]
        embeddings = bert.encode(all_text, show_progress_bar=True, convert_to_tensor=True,
                                 normalize_embeddings=True, batch_size=1024)
        embeddings = embeddings.cpu()
        torch.save((all_text, embeddings), cache_filename)
        print("Saved to ", cache_filename)

        return {text: emb for text, emb in zip(all_text, embeddings)}

    def __getitem__(self, index):
        result = super(SubgraphFewshotDatasetWithTextFeats, self).__getitem__(index)
        support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, \
        query_subgraphs, negative_triples, negative_subgraphs, curr_rel = result
        support_subgraphs = [self._add_text_feats_to_pyg(data) for data in support_subgraphs]
        support_negative_subgraphs = [self._add_text_feats_to_pyg(data) for data in support_negative_subgraphs]
        query_subgraphs = [self._add_text_feats_to_pyg(data) for data in query_subgraphs]
        negative_subgraphs = [self._add_text_feats_to_pyg(data) for data in negative_subgraphs]
        return support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, \
               query_subgraphs, negative_triples, negative_subgraphs, curr_rel

    def next_one_on_eval(self, index):
        result = super(SubgraphFewshotDatasetWithTextFeats, self).next_one_on_eval(index)
        support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, \
        query_subgraphs, negative_triples, negative_subgraphs, curr_rel = result
        support_subgraphs = [self._add_text_feats_to_pyg(data) for data in support_subgraphs]
        support_negative_subgraphs = [self._add_text_feats_to_pyg(data) for data in support_negative_subgraphs]
        query_subgraphs = [self._add_text_feats_to_pyg(data) for data in query_subgraphs]
        negative_subgraphs = [self._add_text_feats_to_pyg(data) for data in negative_subgraphs]
        return support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, \
               query_subgraphs, negative_triples, negative_subgraphs, curr_rel

 
