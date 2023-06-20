# See sample commands at the bottom of this script

pretrain_cmds = {
    "PRODIGY":      "python3 experiments/run_single_experiment.py --root {root} --dataset Wiki --emb_dim 256 --device {device} --input_dim 768 --workers 7 --layers S2,UX,M2 -ds_cap 10010 -lr 1e-3 --prefix Wiki_PT_PRODIGY     -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way 15 -bs 10 -qry 4 -shot 3 --task cls_nm                  --ignore_label_embeddings True -val_cap 10 -test_cap 10 -aug ND0.5,NZ0.5 -aug_test True -ckpt_step 1000 --all_test True -attr 1000 -meta_pos True",
    "Contrastive":  "python3 experiments/run_single_experiment.py --root {root} --dataset Wiki --emb_dim 256 --device {device} --input_dim 768 --workers 7 --layers S2,UX,A  -ds_cap 10010 -lr 1e-3 --prefix Wiki_PT_Contrastive -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way 15 -bs 10 -qry 4 -shot 1 --task same_graph              --ignore_label_embeddings True -val_cap 10 -test_cap 10 -aug ND0.5,NZ0.5 -aug_test True -ckpt_step 1000 --all_test True",
    "PG-NM":        "python3 experiments/run_single_experiment.py --root {root} --dataset Wiki --emb_dim 256 --device {device} --input_dim 768 --workers 0 --layers S2,UX,M2 -ds_cap 10010 -lr 1e-3 --prefix Wiki_PT_PG_NM       -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way 15 -bs 10 -qry 4 -shot 3 --task neighbor_matching       --ignore_label_embeddings True -val_cap 10 -test_cap 10 -aug ND0.5,NZ0.5 -aug_test True -ckpt_step 1000 --all_test True -attr 1000 -meta_pos True",
    "PG-MT":        "python3 experiments/run_single_experiment.py --root {root} --dataset Wiki --emb_dim 256 --device {device} --input_dim 768 --workers 0 --layers S2,UX,M2 -ds_cap 10010 -lr 1e-3 --prefix Wiki_PT_PG_MT       -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way 15 -bs 10 -qry 4 -shot 3 --task multiway_classification --ignore_label_embeddings True -val_cap 10 -test_cap 10 -aug ND0.5,NZ0.5 -aug_test True -ckpt_step 1000 --all_test True -attr 1000 -meta_pos True",
}


def print_pretrain_commands(device=0, dataset_path="<DATA_ROOT>"):
    for k, v in pretrain_cmds.items():
        print(k)
        print(v.format(device=device, root=dataset_path))


n_rels = {
    "NELL": 291,
    "ConceptNet": 14,
    "FB15K-237": 200
}

def get_rels(dataset, ways):
    #seed = ways * 62  # change the seed to get different rels
    import random
    #rnd = random.Random(seed)
    rnd = random.Random()
    if dataset == "NELL":
        lbls = [1, 2, 3, 4, 6, 9, 10, 11, 13, 14, 16, 18, 19, 22, 25, 27, 29, 31, 32, 35, 38, 39, 42, 45, 46, 51, 55, 57,
                59, 60, 62, 63, 66, 69, 70, 71, 73, 77, 78, 79, 82, 84, 88, 90, 91, 92, 93, 94, 102, 105, 106, 107, 108,
                109, 110, 112, 115, 116, 120, 121, 122, 123, 126, 127, 128, 129, 130, 136, 143, 152, 155, 157, 158, 159,
                160, 168, 170, 171, 173, 175, 176, 177, 178, 181, 183, 186, 187, 188, 189, 190, 192, 194, 195, 198, 202,
                204, 206, 209, 212, 215, 217, 219, 220, 221, 225, 227, 230, 231, 236, 240, 253, 255, 256, 257, 258, 259,
                261, 262, 263, 264, 266, 267, 273, 274, 276, 277, 279, 281, 282, 283, 285, 286, 289, 290]
    else:
        lbls = list(range(n_rels[dataset]))
    return rnd.sample(lbls, ways)


linear_probe_cmd = \
"""python3 experiments/run_single_experiment.py
--root {root} --dataset {ds} --emb_dim 256 --device {device} --input_dim 768
--layers S2,UX -ds_cap 1010 --prefix {prefix} -esp 50 --eval_step 200 
--epochs 1 --dropout 0 --n_way {nway} -bs 10 -qry 4 --ignore_label_embeddings True
-lr 1e-4 --task multiway_classification -test_cap 100 -val_cap 20 --workers 0
-meta_pos True {pt_suffix} -ckpt_step 200 --no_split_labels True
--label_set {label_set} -train_cap 10 --linear_probe True"""

ways = {
    "NELL": [40, 20, 10, 5],
    "FB15K-237": [40, 20, 10, 5],
    "ConceptNet": [4]
}

def print_linear_probe_commands(device=0, dataset_path="<DATA_ROOT>", dataset="NELL", pretrained_model=""):
    pt_suffix = ""
    if pretrained_model != "":
        pt_suffix = " -pretrained " + pretrained_model
    for w in ways[dataset]:
        print(linear_probe_cmd.format(root=dataset_path, ds=dataset, device=device, prefix=dataset + "_LinearProbe", pt_suffix=pt_suffix, nway=w, label_set=" ".join([str(x) for x in get_rels(dataset, w)])).replace("\n", " "))



in_context_learning_eval_runs = [
    ["<PATH_TO_PRODIGY_CHECKPOINT>", "S2,UX,M2", "InContext_eval_PRODIGY"],
    # ["<PATH_TO_PG_MT_CHECKPOINT>", "S2,UX,M2", "InContext_eval_PG_MT"],
    # ["<PATH_TO_PG_NM_CHECKPOINT>", "S2,UX,M2", "InContext_eval_PG_NM"],
    # ["<PATH_TO_Contrastive_CHECKPOINT>", "S2,UX,A", "InContext_eval_Contrastive"],
    # ["", "S2,UX,M2", "InContext_eval_NoPretrain"]
]

def get_suffix_lblsplit(dataset):
    lbls = list(range(n_rels[dataset]))
    if dataset == "NELL":
        # filtered out ones that can't do 50shot on NELL
        lbls = [1, 2, 3, 4, 6, 9, 10, 11, 13, 14, 16, 18, 19, 22, 25, 27, 29, 31, 32, 35, 38, 39, 42, 45, 46, 51, 55, 57, 59, 60, 62, 63, 66, 69, 70, 71, 73, 77, 78, 79, 82, 84, 88, 90, 91, 92, 93, 94, 102, 105, 106, 107, 108, 109, 110, 112, 115, 116, 120, 121, 122, 123, 126, 127, 128, 129, 130, 136, 143, 152, 155, 157, 158, 159, 160, 168, 170, 171, 173, 175, 176, 177, 178, 181, 183, 186, 187, 188, 189, 190, 192, 194, 195, 198, 202, 204, 206, 209, 212, 215, 217, 219, 220, 221, 225, 227, 230, 231, 236, 240, 253, 255, 256, 257, 258, 259, 261, 262, 263, 264, 266, 267, 273, 274, 276, 277, 279, 281, 282, 283, 285, 286, 289, 290]
    lbls = " ".join([str(l) for l in lbls])
    return f" --no_split_labels True  --label_set {lbls}  "


def print_in_context_learning_evaluation_cmds(device=0, dataset_path="<DATA_ROOT>", dataset="NELL", n_shots=3):
    cmd_template = "python3 experiments/run_single_experiment.py --root {dataset_path}  --dataset {ds} --emb_dim 256 -shot {nshot} --device {device} --input_dim 768 --layers {layers}  -ds_cap 10 --prefix {prefix} -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way {nway} -bs 1 -qry 4 --ignore_label_embeddings True --task multiway_classification -test_cap 500 -val_cap 10 --workers 15  {pos_only_msg}   --eval_only True {pt_suffix} --all_test True  {suffix_lblsplit}"
    if dataset == "ConceptNet":
        nways = [4]
    elif dataset == "FB15K-237":
        nways = [20] # [40, 20, 10, 5]
    else:
        nways = [20] # [40, 20, 10, 5]
    for run in in_context_learning_eval_runs:
        for nway in nways:
            ignore_lbl_embs = "--ignore_label_embeddings True"
            pos_only_msg = "-meta_pos  True"
            pretrain_path = run[0]
            layers = run[1]
            if pretrain_path != "":
                pretrain_path = pretrain_path.strip()
                pretrain_path = " -pretrained " + pretrain_path
            prefix = run[2]
            s = get_suffix_lblsplit(dataset)
            cmd = cmd_template.format(ds=dataset,
                                        dataset_path = dataset_path,
                                        device=device,
                                        layers=layers,
                                        prefix=prefix,
                                        nway=nway,
                                        pos_only_msg=pos_only_msg,
                                        pt_suffix=pretrain_path,
                                        ignore_lbl_embs=ignore_lbl_embs,
                                        nshot=n_shots,
                                        suffix_lblsplit=s)
            print(cmd)


##### Sample commands #####


# Pretraining - all 4 modes
print_pretrain_commands(device=3, dataset_path="<DATA_ROOT>")

for dataset in ["ConceptNet", "FB15K-237", "NELL"]:
    print_in_context_learning_evaluation_cmds(device=8, dataset_path="<DATA_ROOT>", dataset=dataset, n_shots=3)


# # Uncomment for Linear probe with contrastive checkpoint and limited training data
# for dataset in ways.keys():
#     print_linear_probe_commands(device=8, dataset=dataset, pretrained_model="<PATH_TO_CHECKPOINT>")

# # # Uncomment for all In-context learning evaluation
# for dataset in ways.keys():
#     print_in_context_learning_evaluation_cmds(device=8, dataset_path="<DATA_ROOT>", dataset=dataset, n_shots=3)


