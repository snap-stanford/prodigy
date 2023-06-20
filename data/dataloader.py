import random
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from itertools import chain
import random
from .augment import Identity
import math

class TaskBase:
    def get_label(self, graph_id):
        raise NotImplementedError

    def sample(self, num_label, num_member, rng):
        raise NotImplementedError


class IsomorphismTask(TaskBase):
    def __init__(self, ids):
        """
        `ids` may be a list or a range object, specifying IDs of graphs
        """
        self.ids = ids

    def get_label(self, graph_id):
        return graph_id

    def sample(self, num_label, num_member, rng):
        labels = rng.sample(self.ids, num_label)
        return {label: [label] * num_member for label in labels}

class MultiTaskSplitWay(TaskBase):
    def __init__(self, tasks, task_names, split="even"):
        self.tasks = tasks
        self.task_names = task_names
        self.split = split

    def get_label(self, graph_id):
        raise NotImplementedError

        for task in self.tasks:
            label = task.get_label(graph_id)
            if label is not None:
                return label
        return None

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        labels = {}
        # Evenly get num_label from tasks, distributed, so sum of num_label from each task is num_label
        # ensure that sum of `num_label_task_list` is num_label

        if self.split == "even":
            num_label_task_list = [num_label // len(self.tasks)] * len(self.tasks)
            for i in range(num_label % len(self.tasks)):
                num_label_task_list[i] += 1
            random.shuffle(num_label_task_list)
        elif self.split == "random":
            # Randomly sample num_label from each task, such that sum of num_label from each task is num_label
            num_label_task_list = [rng.randint(1, num_label) for _ in range(len(self.tasks))]
            total = sum(num_label_task_list)
            num_label_task_list = [int(x * num_label / total) for x in num_label_task_list]

            # I have a feeling there will be some off by one error here
            if sum(num_label_task_list) < num_label:
                # print("Warning: sum of num_label_task_list is less than num_label")
                diff = num_label - sum(num_label_task_list)
                # Distribute the rest round robin
                for i in range(diff):
                    num_label_task_list[i % len(num_label_task_list)] += 1
            assert sum(num_label_task_list) == num_label

        else:
            raise ValueError("Unknown split type: {}".format(self.split))

        for task_name, task, num_label_task in zip(self.task_names, self.tasks, num_label_task_list):
            sampled_task_dct = task.sample(num_label_task, num_member, num_shot, num_query, rng)
            for k, v in sampled_task_dct.items():
                labels[(k, task_name)] = v
        return labels


class MultiTaskSplitBatch(TaskBase):
    def __init__(self, tasks, task_names, task_counts):
        self.tasks = tasks
        self.task_names = task_names
        self.task_idx = [i for i, c in enumerate(task_counts) for _ in range(c)]
        random.shuffle(self.task_idx)
        self.task_idx_idx = 0
        self.rng = random.Random()

    def get_label(self, graph_id):
        raise NotImplementedError

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        # Sample randomly from rng to pick which task
        task_idx = self.task_idx[self.task_idx_idx]
        self.task_idx_idx = (self.task_idx_idx + 1) % len(self.task_idx)
        task = self.tasks[task_idx]
        task_name = self.task_names[task_idx]
        sampled_task_dct = task.sample(num_label, num_member, num_shot, num_query, rng)
        labels = {}
        for k, v in sampled_task_dct.items():
            labels[(k, task_name)] = v
        return labels

class MulticlassTask(TaskBase):
    def __init__(self, labels, label_set, train_label=None, linear_probe=False):
        """Multi-class classification
        `labels` is a numpy array
        `label_set` is the set of labels we interested in
        """
        self.labels = labels
        self.label_set = label_set
        self.train_label = train_label
        self.linear_probe = linear_probe
        self.label2idx = {label: np.where(labels == label)[0] for label in label_set}
        if train_label is not None:
            self.train_label2idx = {label: np.where(train_label == label)[0] for label in label_set}
            print(self.train_label2idx )


    def get_label(self, graph_id):
        return self.labels[graph_id].item()

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        if self.linear_probe:
            labels = self.label_set
            assert len(labels) == num_label
        else:
            labels = rng.sample(self.label_set, num_label)

        task = {}
        if self.train_label is None:
            for label in labels:
                members = self.label2idx[label]
                if members.shape[0] < num_member:
                    sample_func = rng.choices
                else:
                    sample_func = rng.sample
                task[label] = members[sample_func(range(members.shape[0]), k=num_member)].tolist()
        else:
            for label in labels:
                members = self.label2idx[label]
                train_members = self.train_label2idx[label]
                #if members.shape[0] == 0:
                #    continue
                if members.shape[0] < num_query:
                    sample_func = rng.choices
                else:
                    sample_func = rng.sample
                if train_members.shape[0] < num_shot:
                    train_sample_func = rng.choices
                else:
                    train_sample_func = rng.sample
                task[label] = train_members[train_sample_func(range(train_members.shape[0]), k=num_shot)].tolist() + members[sample_func(range(members.shape[0]), k=num_query)].tolist()
        return task

class ContrastiveTask(TaskBase):
    def __init__(self, size):
        self.size = size

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        # Task is dict
        # Key: center node (int)
        # Value: list of nodes (list of int), list of nodes is the members of the task
        task = {}
        while len(task) < num_label:
            center = rng.randrange(self.size)
            if center in task:
                continue
            node_idx = torch.ones(num_member, dtype=torch.long) * center
            task[center] = node_idx.tolist()

        return task

class NeighborTask(TaskBase):
    def __init__(self, neighbor_sampler, size, direction):
        self.neighbor_sampler = neighbor_sampler
        self.size = size
        self.direction = direction

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        # Task is dict
        # Key: center node (int)
        # Value: list of nodes (list of int), list of nodes is the members of the task
        task = {}
        while len(task) < num_label:
            center = rng.randrange(self.size)
            if center in task:
                continue
            node_idx = torch.ones(num_member * 10, dtype=torch.long) * center
            node_idx = self.neighbor_sampler.random_walk(node_idx, self.direction)
            node_idx = torch.unique(node_idx)
            if node_idx.size(0) >= num_member:
                task[center] = node_idx[:num_member].tolist()

        return task

class KGNeighborTask(TaskBase):
    def __init__(self, dataset, neighbor_sampler, size, direction, is_multiway):
        self.dataset = dataset
        self.neighbor_sampler = neighbor_sampler
        self.size = size
        self.direction = direction
        self.is_multiway = is_multiway

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        # Task is dict
        # Key: center node (int)
        # Value: list of nodes (list of int), list of nodes is the members of the task
        task = {}
        while len(task) < num_label:
            center = rng.randrange(self.size)
            if center in task:
                continue
            node_idx = torch.ones(num_member * 10, dtype=torch.long) * center
            node_idx = self.neighbor_sampler.random_walk(node_idx, self.direction)
            node_idx = torch.unique(node_idx)
            # edge_idx = self.dataset.sample_subgraph_around_node(node_idx)
            edge_idx = self.neighbor_sampler.sample_edge(node_idx, "inout")
            edge_idx = torch.unique(edge_idx)
            if edge_idx.size(0) >= num_member:
                task[center] = edge_idx[:num_member].tolist()
                if not self.is_multiway and len(task) == 1:
                    num_member = 1

        return task

@dataclass
class BatchParam:
    batch_size: int
    n_way: int
    n_shot: int
    n_query: int
    n_aug: int
    n_member: int


class ParamSampler:
    def __init__(self, batch_size, n_way, n_shot, n_query, n_aug):
        self.batch_size = batch_size
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_aug = n_aug

    @staticmethod
    def sample_param(dist, rng):
        if isinstance(dist, int):
            return dist
        else:
            return rng.choice(dist)

    def __call__(self, rng):
        batch_size = self.sample_param(self.batch_size, rng)

        n_way = self.sample_param(self.n_way, rng)
        n_shot = self.sample_param(self.n_shot, rng)
        n_query = self.sample_param(self.n_query, rng)
        n_aug = self.sample_param(self.n_aug, rng)
        n_member = (n_shot + n_query) // n_aug

        return BatchParam(
            batch_size=batch_size,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_aug=n_aug,
            n_member=n_member,
        )


class BatchSampler(Sampler):
    """Sample fewshot tasks for meta-learning."""
    def __init__(self,
        num_samples,
        task,
        param_sampler,
        seed=None,
    ):
        self.num_samples = num_samples
        self.task = task
        self.param_sampler = param_sampler
        self.rng = random.Random(seed)

    def __iter__(self):
        for i in range(self.num_samples):
            yield self.sample()

    def __len__(self):
        return self.num_samples

    def sample(self):
        batch_param = self.param_sampler(self.rng)

        batch = []
        for _ in range(batch_param.batch_size):
            batch.append(self.task.sample(batch_param.n_way, batch_param.n_member, batch_param.n_shot, batch_param.n_query, self.rng))
        return batch, batch_param


def linearize(mask, inputs_idx, output_idx, batch_rand_perm = None):
    if batch_rand_perm is None:
        rand = torch.rand(len(mask), mask[0][mask[0]].size(0))
        batch_rand_perm = rand.argsort(dim=1)

    inputs_idx = inputs_idx[mask].reshape(len(mask), - 1)
    output_idx = output_idx[mask].reshape(len(mask), - 1)
    inputs_idx = torch.take_along_dim(inputs_idx, batch_rand_perm, 1)
    output_idx = torch.take_along_dim(output_idx, batch_rand_perm, 1)
    seqs = torch.stack([inputs_idx, output_idx], dim=1)

    return seqs.transpose(2,1).reshape(seqs.shape[0], -1), batch_rand_perm

class Collator:
    def __init__(self, label_meta, aug=Identity(), is_multiway=True):
        self.label_meta = label_meta
        self.aug = aug
        self.is_multiway = is_multiway

    def process_one_task(self, task, batch_param):
        label_map = list(task) # Looks like this: (0, 'task1'), (1, 'task2'), ...
        label_map_reverse = {v: i for i, v in enumerate(label_map)} # ((0, 'task1'), 0), ((1, 'task2'), 1), ...
        all_graphs = []
        labels = []
        query_mask = []
        for label, graphs in task.items():
            augmented = [self.aug(graph) for graph in graphs for _ in range(batch_param.n_aug)]
            all_graphs.extend(augmented)
            query_mask.extend([False] * (batch_param.n_shot))
            query_mask.extend([True] * (len(augmented) - batch_param.n_shot))
            labels.extend([label_map_reverse[label]] * len(augmented)) # label_map_reverse[label] is the index of label in label_map
        return all_graphs, torch.tensor(labels), torch.tensor(query_mask), label_map

    def __call__(self, batch):
        batch, batch_param = batch

        # batch is [batch_idx][support or query][label][instance_idx] -> sampled subgraph
        graphs, labels, query_mask, label_map = map(list, zip(*[self.process_one_task(task, batch_param) for task in batch]))
        # import pdb;pdb.set_trace()
        # print([ g.center_node_idx  for g in graphs[0]])
        num_task = len(graphs)
        task_len = len(graphs[0])
        assert all(len(i) == task_len for i in graphs)
        num_labels = len(label_map[0])
        assert all(len(i) == num_labels for i in label_map) # label_map length is the same for all tasks

        graphs = Batch.from_data_list([g for l in graphs for g in l])
        labels = torch.cat(labels)
        b_mask = torch.stack(query_mask)
        query_mask = torch.cat(query_mask)
        label_map = list(chain(*label_map))
        if self.is_multiway:
            metagraph_edge_source = torch.arange(labels.size(0)).repeat_interleave(num_labels)
            metagraph_edge_target = torch.arange(num_labels).repeat(labels.size(0))
            metagraph_edge_target += (torch.arange(num_task) * num_labels).repeat_interleave(task_len * num_labels) + labels.size(0)
            metagraph_edge_index = torch.stack([metagraph_edge_source, metagraph_edge_target], dim=0)

            metagraph_edge_mask = query_mask.repeat_interleave(num_labels)
            metagraph_edge_attr = torch.nn.functional.one_hot(labels, num_labels).float().reshape(-1)
            metagraph_edge_attr = (metagraph_edge_attr * 2 - 1) * (~metagraph_edge_mask)
            metagraph_edge_attr = torch.stack([metagraph_edge_mask, metagraph_edge_attr], dim=1)

            # Tuple case, where the first element is the label id and the second element is the task name.
            if isinstance(label_map[0], tuple) and len(label_map[0]) == 2:
                label_embeddings = []
                for (label_id, label_name) in label_map:
                    label_embeddings.append(self.label_meta[label_name][label_id])
                label_embeddings = torch.stack(label_embeddings, dim=0)
            else:
                label_map = torch.tensor(label_map)
                label_embeddings = self.label_meta[label_map]
            labels_onehot = torch.nn.functional.one_hot(labels).float()

            a = metagraph_edge_index[:, labels_onehot.flatten() == 1]
            inputs_idx = a.reshape(2, len(b_mask), -1)[0]
            output_idx = a.reshape(2, len(b_mask), -1)[1]


            input_seqs, _ = linearize(~b_mask, inputs_idx, output_idx)
            query_seqs, batch_rand_perm = linearize(b_mask, inputs_idx, torch.ones(output_idx.shape, dtype=torch.int) * (metagraph_edge_index.max()+ 1))
            query_seqs_gt, _ = linearize(b_mask, inputs_idx, output_idx, batch_rand_perm)
            return graphs, label_embeddings, labels_onehot, metagraph_edge_index, metagraph_edge_attr, metagraph_edge_mask, input_seqs, query_seqs, query_seqs_gt


        else:
            metagraph_edge_source = torch.arange(labels.size(0))
            metagraph_edge_target = torch.arange(1).repeat(labels.size(0))  # we have just one label
            metagraph_edge_target += labels.size(0) + (torch.arange(num_task)).repeat_interleave(task_len)
            metagraph_edge_index = torch.stack([metagraph_edge_source, metagraph_edge_target], dim=0)
            metagraph_edge_mask = query_mask
            metagraph_edge_attr = labels.float().reshape(-1)
            metagraph_edge_attr = (metagraph_edge_attr * 2 - 1) * (~metagraph_edge_mask)
            metagraph_edge_attr = torch.stack([metagraph_edge_mask, metagraph_edge_attr], dim=1)
            label_map = label_map[1::2]
            # Tuple case, where the first element is the label id and the second element is the task name.
            if isinstance(label_map[0], tuple) and len(label_map[0]) == 2:
                label_embeddings = []
                for (label_id, label_name) in label_map:
                    label_embeddings.append(self.label_meta[label_name][label_id])
                label_embeddings = torch.stack(label_embeddings, dim=0)
            else:
                label_map = torch.tensor(label_map)
                label_embeddings = self.label_meta[label_map]
            #labels_onehot = torch.nn.functional.one_hot(labels).float()
            labels_onehot = labels.float().reshape(-1,1)

            inputs_idx = metagraph_edge_index.reshape(2, len(b_mask), -1)[0]
            output_idx = metagraph_edge_index.reshape(2, len(b_mask), -1)[1].clone()

            # add a fake False class for binary classification
            output_idx[labels_onehot.reshape(len(b_mask), -1) == 0] = metagraph_edge_index.max() + 2

            input_seqs, _ = linearize(~b_mask, inputs_idx, output_idx)
            query_seqs, batch_rand_perm = linearize(b_mask, inputs_idx, torch.ones(output_idx.shape, dtype=torch.int) * (
                        metagraph_edge_index.max() + 1))
            query_seqs_gt, _ = linearize(b_mask, inputs_idx, output_idx, batch_rand_perm)
            return graphs, label_embeddings, labels_onehot, metagraph_edge_index, metagraph_edge_attr, metagraph_edge_mask, input_seqs, query_seqs, query_seqs_gt


class KGCollator(Collator):
    def __init__(self, label_meta, aug=Identity(), is_multiway=True):
        super(KGCollator, self).__init__(label_meta, aug, is_multiway)

    def process_one_task(self, task, batch_param):
        label_map = list(task)
        if not self.is_multiway:
            new_tasks = {}
            pos_idx = label_map[0]
            new_tasks[pos_idx] = task[pos_idx]
            # create a mixed up negative task
            all_graphs = []
            for label, graphs in task.items():
                if label == pos_idx:
                    continue
                all_graphs.append(graphs[0])
            new_tasks[-1] =all_graphs
            task = new_tasks
            label_map = [-1, pos_idx]

        # Looks like this: (0, 'task1'), (1, 'task2'), ...
        label_map_reverse = {v: i for i, v in enumerate(label_map)} # ((0, 'task1'), 0), ((1, 'task2'), 1), ...
        all_graphs = []
        labels = []
        query_mask = []
        for label, graphs in task.items():
            augmented = [self.aug(graph) for graph in graphs for _ in range(batch_param.n_aug)]
            all_graphs.extend(augmented)
            query_mask.extend([False] * (batch_param.n_shot))
            query_mask.extend([True] * (len(augmented) - batch_param.n_shot))
            labels.extend([label_map_reverse[label]] * len(augmented)) # label_map_reverse[label] is the index of label in label_map
        for data in all_graphs:
            data.x = torch.cat([data.x, torch.zeros(data.x.shape[0], 2)], dim=1)
            data.x[0, -1] = 1.
            data.x[1, -2] = 1.  # flag the head and tail nodes
            if hasattr(data, "x_orig"):
                data.x_orig = torch.cat([data.x_orig, torch.zeros(data.x_orig.shape[0], 2)], dim=1)
                data.x_orig[0, -1] = 1.
                data.x_orig[1, -2] = 1.  # flag the head and tail nodes
        return all_graphs, torch.tensor(labels), torch.tensor(query_mask), label_map
