import torch
import numpy as np
import sys
import os
import wandb
import torch.optim as optim
import time
from tqdm import tqdm, trange
import shutil

sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))

from models.get_model import print_num_trainable_params
from models.model_eval_utils import accuracy
from models.general_gnn import SingleLayerGeneralGNN
from models.simple_dot_product import SimpleDotProdModel
from models.sentence_embedding import SentenceEmb
from experiments.layers import get_module_list


class TrainerFS():
    def __init__(self, dataset, parameter):
        wandb.init(project="graph-clip", name=parameter["exp_name"])
        #wandb.run.log_code(".")
        wandb.run.summary["wandb_url"] = wandb.run.url
        print("---------Parameters---------")
        for k, v in parameter.items():
            print(k + ': ' + str(v))
        print("----------------------------")
        wandb.config.trainer_fs = True

        self.parameter = parameter

        self.ignore_label_embeddings = parameter['ignore_label_embeddings']
        self.is_zero_shot = parameter['zero_shot']

        # parameters
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.dataset_len_cap = parameter['dataset_len_cap']
        self.invalidate_cache = parameter['invalidate_cache']
        self.early_stopping_patience = parameter['early_stopping_patience']

        # step
        self.steps = parameter["epochs"] * parameter['dataset_len_cap']
        self.print_step = parameter['print_step']
        self.eval_step = parameter['eval_step']
        self.checkpoint_step = parameter['checkpoint_step']

        self.dataset_name = parameter['dataset']
        self.classification_only = self.parameter["classification_only"]

        self.shots = parameter['n_shots']  # k shots!
        self.ways = parameter['n_way']  # n way classification!

        self.device = parameter['device']

        if self.ways > 1:
            self.loss = torch.nn.CrossEntropyLoss()
            self.is_multiway = True
        elif self.ways == 1:
            self.loss = torch.nn.BCEWithLogitsLoss()  # binary classification (positives/negatives)
            self.is_multiway = False
        else:
            raise Exception("Invalid number of ways:", self.ways)

        self.calc_ranks = parameter['calc_ranks']
        self.cos = torch.nn.CosineSimilarity(dim=1)

        bert_dim = 768

        self.emb_dim = parameter["emb_dim"]
        self.gnn_type = parameter["gnn_type"]
        self.original_features = parameter["original_features"]

        self.fix_datasets = self.parameter['fix_datasets_first']


        initial_label_mlp = torch.nn.Linear(bert_dim, self.emb_dim)
                                              
        edge_attr_dim = None
        if self.dataset_name in ["NELL", "ConceptNet", "FB15K-237", "Wiki", "WikiKG90M"]:
            edge_attr_dim = bert_dim
            self.parameter["input_dim"] = bert_dim + 2  # add 2 to flag head and tail nodes
            if self.parameter["task_name"] == "neighbor_matching":
                edge_attr_dim = bert_dim
            if self.parameter["task_name"] == "sn_neighbor_matching":
                edge_attr_dim = bert_dim
                self.parameter["input_dim"] = bert_dim
            if self.parameter["kg_emb_model"]:
                # if KG embedding model is set, we ignore the input_dim kwarg
                kg_embedding_dim = 100
                edge_attr_dim = kg_embedding_dim
                self.parameter["input_dim"] = kg_embedding_dim + 2  # add 2 to flag head and tail nodes
        if self.dataset_name in ["CSG"]:
            edge_attr_dim = 128

        self.txt_dropout = torch.nn.Dropout(self.parameter["text_features_dropout"])
        self.msg_pos_only = "meta_gnn_pos_only" in self.parameter and self.parameter["meta_gnn_pos_only"]
        if self.parameter["layers"] != "SimpleDotProduct":
            batch_norm_encoder = not self.parameter["no_bn_encoder"]
            batch_norm_metagraph = not self.parameter["no_bn_metagraph"]
            layer_list = get_module_list(self.parameter["layers"], self.emb_dim, edge_attr_dim=edge_attr_dim,
                                         input_dim=self.parameter["input_dim"], dropout=self.parameter["dropout"],
                                         reset_after_layer = self.parameter["reset_after_layer"],
                                         attention_mask_scheme = self.parameter["attention_mask_scheme"],
                                         has_final_back = self.parameter["has_final_back"],
                                         msg_pos_only=self.msg_pos_only,
                                         batch_norm_metagraph=batch_norm_metagraph,
                                         batch_norm_encoder=batch_norm_encoder,
                                         gnn_use_relu = self.dataset_name in ["NELL", "ConceptNet", "FB15K-237", "Wiki", "WikiKG90M"])

            layer_list = torch.nn.ModuleList(layer_list)
            self.model = SingleLayerGeneralGNN(layer_list=layer_list, initial_label_mlp=initial_label_mlp,  # initial_input_mlp = initial_input_mlp,
                                                 params=self.parameter, text_dropout=self.txt_dropout)
        else:
            self.model = SimpleDotProdModel(layer_list=None, initial_label_mlp=initial_label_mlp,
                                            params=self.parameter, text_dropout=self.txt_dropout)
        print(self.model)
        self.model.to(self.device)
        num_params = print_num_trainable_params(self.model)
        # Add logging of # params to summary.json
        wandb.run.summary["num_params"] = num_params

        # create a header to predict masked node attribute
        if self.parameter["attr_regression_weight"]:
            embed_dim = self.emb_dim
            output_dim = self.parameter["input_dim"]
            self.aux_header = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim, output_dim),
            )
            self.aux_header.to(self.device)
            self.aux_loss = torch.nn.MSELoss()
            self.aux_loss.to(self.device)

        bert_model_name = self.parameter["bert_emb_model"]
        self.Bert = SentenceEmb(bert_model_name, device=self.device, cache_folder=os.path.join(self.parameter["root"], "sbert"))

        params = list(self.model.parameters())
        if hasattr(self, "aux_header"):
            params += list(self.aux_header.parameters())
        if not self.parameter["not_freeze_learned_label_embedding"]:
            for param in self.model.learned_label_embedding.parameters():
                param.requires_grad = False

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, params),
                                     lr=self.learning_rate, weight_decay=self.parameter["weight_decay"])

        # self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, 0, self.steps)

        wandb.config.params = parameter
        wandb.watch(self.model, log_freq=100)

        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['exp_name'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        # Symlink to latest checkpoint
        self.wandb_fdir = os.path.join(self.state_dir, 'files')
        if not os.path.isdir(self.wandb_fdir):
            os.symlink(wandb.run.dir, self.wandb_fdir)

        self.ckpt_dir = os.path.join(self.state_dir, 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        self.logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['exp_name'], 'data')
        self.cache_dir = os.path.join(self.parameter['log_dir'], "cache")
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        if not os.path.isdir(self.logging_dir):
            os.makedirs(self.logging_dir)
        else:
            if self.parameter["override_log"]:
                print(f"Overwriting {self.logging_dir} logging dir!")
                shutil.rmtree(self.logging_dir)
                os.makedirs(self.logging_dir)
            else:
                raise Exception(f"{self.logging_dir} logging dir already exists!!!")

        self.all_saveable_modules = {
            "model": self.model
        }
        self.pretrained_model_run = self.parameter["pretrained_model_run"]
        if self.pretrained_model_run != "":
            print("Reload state dict from path", self.pretrained_model_run)
            self.load_checkpoint(self.pretrained_model_run)

        # Data loader creation.
        self.train_dataloader, self.train_val_dataloader, self.val_dataloader, self.test_dataloader = self._build_dataloaders(dataset, self.dataset_name)

    def _build_dataloaders(self, dataset, dataset_name):
        kwargs = {}
        kwargs["root"] = os.path.join(self.parameter["root"], dataset_name)
        kwargs["num_workers"] = self.parameter["workers"]
        kwargs["batch_size"] = self.parameter["batch_size"]
        kwargs["n_way"] = self.parameter["n_way"]
        kwargs["n_shot"] = self.parameter["n_shots"]
        kwargs["n_query"] = self.parameter["n_query"]
        kwargs["bert"] = self.Bert
        kwargs["task_name"] = self.parameter["task_name"]
        kwargs["aug"] = self.parameter["augmentation"]
        kwargs["aug_test"] = self.parameter["augment_test"]
        kwargs["split_labels"] = not self.parameter["no_split_labels"]
        kwargs["train_cap"] = self.parameter["train_cap"]
        kwargs['linear_probe'] = self.parameter['linear_probe']
        if self.parameter["all_test"]:
            kwargs["all_test"] = True
        if self.parameter["label_set"]:
            kwargs["label_set"] = set([int(v) for v in self.parameter["label_set"]])
            print("Label set:", kwargs["label_set"])
        if self.parameter["csr_split"]:
            kwargs["csr_split"] = self.parameter["csr_split"]
        if dataset_name == "arxiv":
            from data.arxiv import get_arxiv_dataloader
            get_dataloader = get_arxiv_dataloader
        elif dataset_name == "mag240m":
            from data.mag240m import get_mag240m_dataloader
            get_dataloader = get_mag240m_dataloader
        elif dataset_name in ["Wiki", "WikiKG90M"]: # "NELL", "FB15K-237", "ConceptNet",  by default still use legacy for them for now
            from data.kg import get_kg_dataloader
            get_dataloader = get_kg_dataloader
        elif dataset_name in [ "NELL", "FB15K-237", "ConceptNet"]: 
            assert self.parameter["task_name"] != "classification"
            from data.kg import get_kg_dataloader
            get_dataloader = get_kg_dataloader
        else:
            raise NotImplementedError

        val_dataloader = get_dataloader(dataset, split="val", node_split="", batch_count=self.parameter["val_len_cap"], **kwargs)
        test_dataloader = get_dataloader(dataset, split="test", node_split="", batch_count=self.parameter["test_len_cap"], **kwargs)

        train_val_dataloader = None
        train_node_split = ""
        if self.parameter["split_train_nodes"]:
            train_val_dataloader = get_dataloader(dataset, split="train", node_split="val", batch_count=self.parameter["val_len_cap"], **kwargs)
            train_node_split = "train"

        # Update the n_way, n_shot, n_query parameters with range objects for the dataset
        # This is only done for train
        if self.parameter["n_way_upper"] > 0:
            kwargs["n_way"] = range(kwargs["n_way"], self.parameter["n_way_upper"] + 1)
        if self.parameter["n_shots_upper"] > 0:
            kwargs["n_shot"] = range(kwargs["n_shot"], self.parameter["n_shots_upper"] + 1)
        if self.parameter["n_query_upper"] > 0:
            kwargs["n_query"] = range(kwargs["n_query"], self.parameter["n_query_upper"] + 1)
        train_dataloader = get_dataloader(dataset, split="train", node_split=train_node_split, batch_count=self.parameter["dataset_len_cap"], **kwargs)
        return train_dataloader, train_val_dataloader, val_dataloader, test_dataloader


    def move_to_device(self, bt_response):
        return tuple([x.to(self.device) for x in bt_response])
        

    def get_loss_and_acc(self, y_true_matrix, y_pred_matrix):
        loss = self.loss(y_pred_matrix, y_true_matrix.float())
        if not self.is_multiway:
            p_score = y_pred_matrix[y_true_matrix == 1]
            n_score = y_pred_matrix[y_true_matrix == 0]
            if len(p_score) == len(n_score):
                y = torch.Tensor([1]).to(y_true_matrix.device)
                loss = torch.nn.MarginRankingLoss(0.5)(p_score, n_score, y)
            else:
                print("Not using ranking loss")

        return loss, accuracy(y_true_matrix, y_pred_matrix, calc_roc=not self.is_multiway)[2]
    
    def get_hits(self, y_true_matrix, y_pred_matrix, task_mask):
        # get HITS@10, HITS@5, HITS@1, MRR scores
        tasks = task_mask.unique()
        n_tasks = len(tasks)
        yt, yp = y_true_matrix.cpu().numpy().flatten(), y_pred_matrix.cpu().numpy().flatten()
        data = {"Hits@10": 0, "Hits@5": 0, "Hits@1": 0, "MRR": 0}
        for i in range(n_tasks):
            where = torch.where(task_mask == tasks[i])[0].cpu()
            x = torch.tensor(yp[where])
            query_idx = np.where(yt[where] == 1)[0]
            _, idx = torch.sort(x, descending=True)
            rank = list(idx.cpu().numpy()).index(query_idx) + 1
            if rank <= 10:
                data['Hits@10'] += 1
            if rank <= 5:
                data['Hits@5'] += 1
            if rank == 1:
                data['Hits@1'] += 1
            data['MRR'] += 1.0 / rank
        for key in data:
            data[key] = data[key] / n_tasks
        return data

    def get_aux_loss(self, graph):
        if hasattr(graph, "node_attr_mask") and self.parameter["attr_regression_weight"]:
            mask = ~graph.node_attr_mask
            if hasattr(graph, "node_mask"):
                mask = mask.logical_and(graph.node_mask)
            target = graph.x_orig[mask]
            input = graph.x[mask]
            output = self.aux_header(input)
            loss = self.aux_loss(output, target)
            return loss
        return torch.zeros(1, device=self.device)

    def save_checkpoint(self, step):
        state_dict = {key: value.state_dict() for key, value in self.all_saveable_modules.items()}
        torch.save(state_dict, os.path.join(self.ckpt_dir, 'state_dict_' + str(step) + '.ckpt'))

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        for key, module in self.all_saveable_modules.items():
            module.load_state_dict(state_dict[key], strict=False)


    def save_best_state_dict(self, best_step):
        best_step = os.path.join(self.ckpt_dir, 'state_dict_' + str(best_step) + '.ckpt')
        best_ckpt = os.path.join(self.state_dir, 'state_dict')
        # Check if best_step exists
        if os.path.exists(best_step):
            shutil.copy(best_step, best_ckpt)
        else:
            print('No such best checkpoint to copy: {}'.format(best_step))
        print("Saved best model to {}".format(best_ckpt))
        self.best_state_dict_path = best_ckpt

    def train(self):

        # initialization
        best_step = 0
        best_val = 0
        test_acc_on_best_val = 0
        best_test_acc = 0
        other_metrics_on_best = {}
        bad_counts = 0

        # training by step
        t_load, t_one_step = 0, 0
        pbar = trange(self.steps)
        train_dataloader_itr = iter(self.train_dataloader)

        bad_counts = 0

        def prefix_dict(d, prefix):
            return {prefix + key: value for key, value in d.items()}
        
        with torch.no_grad():
            # self.model.eval()
            test_loss, test_acc, test_acc_std, test_aux_loss, ranks = self.do_eval(self.test_dataloader)
            start_log_dict = {"start_test_acc": test_acc, "start_test_acc_std": test_acc_std}
            if ranks is not None:
                for key in ranks:
                    start_log_dict["start_test_" + key] = ranks[key]
            wandb.log(start_log_dict) # Test accuracy before training (if using e.g. a pretrained model etc.)

        if "eval_only" in self.parameter and self.parameter["eval_only"]:
            print("Evaluation only - skipping training - exiting now")
            print("Note: also skipping evaluation of val set")
            return

        with torch.no_grad():
            # self.model.eval()
            val_loss, val_acc, val_acc_std, val_aux_loss, ranks = self.do_eval(self.val_dataloader)
            start_log_dict = {"start_val_acc": val_acc, "start_val_acc_std": val_acc_std}
            if ranks is not None:
                for key in ranks:
                    start_log_dict["start_val_" + key] = ranks[key]
            wandb.log(start_log_dict)  # Test accuracy before training (if using e.g. a pretrained model etc.)

        for e in pbar:
            self.model.train()

            self.optimizer.zero_grad()

            t1 = time.time()
            try:
                batch = next(train_dataloader_itr)
            except StopIteration:
                train_dataloader_itr = iter(self.train_dataloader)
                batch = next(train_dataloader_itr)
            t2 = time.time()
            batch = [i.to(self.device) for i in batch]
            yt, yp, graph = self.model(*batch) # apply the model
            loss, acc = self.get_loss_and_acc(yt, yp) # get loss
            aux_loss = self.get_aux_loss(graph)
            weight = self.parameter["attr_regression_weight"]
            total_loss = loss + aux_loss * weight
            total_loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            t3 = time.time()
            wandb.log({"step_time": t3 - t2}, step=e)
            wandb.log({"load_time": t2 - t1}, step=e)
            wandb.log({"train_loss": loss, "train_acc": acc, "train_aux_loss": aux_loss, "train_total_loss": total_loss}, step=e)  # loss and acc here are both floats
            t_load += t2 - t1
            t_one_step += t3 - t2
            pbar.set_description("load: %s, step: %s" % (t_load / (e + 1), t_one_step / (e + 1)))

            # print the loss on specific step
            if e % self.print_step == 0:
                # loss_num = loss
                pbar.write(f"Loss: {loss.item()}")
            # save checkpoint on specific step
            if e % self.checkpoint_step == 0 and e != 0:
                pbar.write('Step  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)

            if e % self.eval_step == 0 and e != 0:
                # pbar.write("Evaluating on validation set!")
                with torch.no_grad():
                    self.model.eval()
                    val_loss, val_acc, val_acc_std, val_aux_loss, ranks = self.do_eval(self.val_dataloader)

                if val_acc >= best_val:
                    best_val = val_acc
                    best_step = e
                    bad_counts = 0
                    self.save_checkpoint(best_step)  # save the best checkpoint
                else:
                    pbar.write("Validation loss did not improve now for {} validation checkpoints".format(bad_counts))
                    bad_counts += 1
                    # if bad_counts >= self.early_stopping_patience:
                    #     pbar.write("Early stopping at step {}".format(e))
                    #     break

                pbar.write(f"Validation loss {val_loss} acc {val_acc} aux_loss {val_aux_loss}")
                wandb.log({"valid_loss": val_loss, "valid_acc": val_acc, "valid_aux_loss": val_aux_loss},
                          step=e)

                if self.train_val_dataloader is not None:
                    with torch.no_grad():
                        self.model.eval()
                        tval_loss, tval_acc, tval_acc_std, tval_aux_loss, ranks = self.do_eval(self.train_val_dataloader)
                        wandb.log({"train_val_loss": tval_loss, "train_val_acc": tval_acc, "train_val_aux_loss": tval_aux_loss}, step=e)

                # Also evaluate on test set
                with torch.no_grad():
                    self.model.eval()
                    test_loss, test_acc, test_acc_std, test_aux_loss, ranks = self.do_eval(self.test_dataloader)
                    log_dict = {"test_acc": test_acc, "test_loss": test_loss.cpu().detach().float(), "test_aux_loss": test_aux_loss, "test_acc_std": test_acc_std}
                    #print("Logging", log_dict)
                    #wandb.log(log_dict, step=e)
                    if ranks is not None:
                        ranks_dict = prefix_dict(ranks, "test_")
                        log_dict.update(ranks_dict)
                    wandb.log(log_dict, step=e)
                    best_test_acc = max(best_test_acc, test_acc)
                    if e == best_step:
                        test_acc_on_best_val = test_acc
                        if ranks is not None:
                            other_metrics_on_best = ranks
        print('Training has finished')
        print('\tBest step is {0} | {1} of valid set is {2:.3f}'.format(best_step, "accuracy", best_val))

        print("Best step is", best_step)
        print("Best testing accuracy is", best_test_acc)
        print("Testing accuracy on best val is", test_acc_on_best_val)
        print("Best val accuracy is", best_val)
        wandb.run.summary["best_step"] = best_step
        wandb.run.summary["best_test_acc"] = best_test_acc
        wandb.run.summary["test_acc_on_best_val"] = test_acc_on_best_val
        wandb.run.summary["final_validation_acc"] = best_val
        if other_metrics_on_best is not None:
              for key in other_metrics_on_best:
                  wandb.run.summary["final_test_" + key] = other_metrics_on_best[key]
        self.save_best_state_dict(best_step)
        print('Finish')
        wandb.finish()
        return best_val, test_acc_on_best_val, best_step
        # returns best-val-acc, best-test-acc, best-step

    def do_eval(self, dataloader, eff_len=None):
        # calc_ranks: if True, it will calculate MRR, HITS scores etc.
        torch.set_grad_enabled(False)  # disable gradient calculation
        ranks = None
        if self.calc_ranks:
            ranks = []
        ytrueall, ypredall = None, None
        all_aux_loss = []
        acc_all = []
        for batch in tqdm(dataloader, leave=False):
            batch = [i.to(self.device) for i in batch]
            yt, yp, graph = self.model(*batch)  # apply the model
            if self.calc_ranks:
                assert len(batch) == 10, "Not using the right batch structure; need to include task_mask"
            loss, acc = self.get_loss_and_acc(yt, yp)  # get loss
            acc_all.append(acc)
            aux_loss = self.get_aux_loss(graph)
            if self.calc_ranks:
                task_mask = batch[9]
                query_set_mask = batch[5]
                query_set_mask = torch.where(query_set_mask == 1)[0]
                curr_ranks = self.get_hits(yt, yp, task_mask[query_set_mask])
                ranks.append([curr_ranks, len(task_mask[query_set_mask.unique()])])  # append values and weights

            # If using random sampling as with MultiTaskSplitWay, need to doubly sample labels to avoid shape dim mismatch
            if ytrueall is None:
                ytrueall = yt
                ypredall = yp
            else:
                ytrueall = torch.cat((ytrueall, yt), dim=0)
                ypredall = torch.cat((ypredall, yp), dim=0)
            all_aux_loss.append(aux_loss.item())
        loss_global, acc_global = self.get_loss_and_acc(ytrueall, ypredall)
        acc_batch_std = np.std(acc_all)
        aux_loss_global = sum(all_aux_loss) / len(all_aux_loss)
        torch.set_grad_enabled(True)
        if ranks is not None:
            ranks = {key: np.average([r[0][key] for r in ranks], weights=[r[1] for r in ranks]) for key in ranks[0][0]}
        return loss_global, acc_global, acc_batch_std, aux_loss_global, ranks

