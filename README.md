# PRODIGY: Enabling In-context Learning Over Graphs



A pretraining framework enabling in-context learning over graphs => pretrain graph model and adapt to diverse downstream tasks on unseen graphs without parameter optimization! 

Paper: https://arxiv.org/abs/2305.12600 (short paper accepted at SPIGM @ ICML 2023)

Authors: Qian Huang, Hongyu Ren, Peng Chen, Gregor Kržmanc, Daniel Zeng, Percy Liang, Jure Leskovec

![In-context few-shot prompting over graphs with prompt graph for edge classification in
PRODIGY. ](fig.png)

# Setup
```
pip install -r requirements.txt
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

All datasets should be prepared to individual folders under <DATA_ROOT>. For MAG and arXiv, the datasets will be automatically downloaded and processed to <DATA_ROOT>. In case of memory issue when generating adjacency matrix, we also provide the preprocessed MAG [adjacency matrix](http://snap.stanford.edu/prodigy/mag240m_adj_bi.pt) that should be put under <DATA_ROOT>/mag240m after the ogb download.

For KG, download preprocessed [Wiki](http://snap.stanford.edu/prodigy/Wiki.zip) and [FB15K-237](http://snap.stanford.edu/prodigy/FB15K-237.zip) datasets to <DATA_ROOT>. Download other KG datasets (NELL and ConceptNet) similarly following links in https://github.com/snap-stanford/csr. 



# Pretraining and Evaluation Commands

### PRODIGY pretraining on MAG240M
```
python experiments/run_single_experiment.py --dataset mag240m  --root <DATA_ROOT>  --original_features True -ds_cap 50010 -val_cap 100 -test_cap 100 --epochs 1 -ckpt_step 1000 -layers S2,U,M -lr 3e-4 -way 30 -shot 3 -qry 4 -eval_step 1000 -task cls_nm_sb  -bs 1 -aug ND0.5,NZ0.5 -aug_test True -attr 1000 --device 0 --prefix MAG_PT_PRODIGY
```

Prefix specifies the run name prefix in wandb and checkpoints will be saved to ./state/MAG_PT_PRODIGY_<time_stamp>/checkpoint/


### PRODIGY evaluation on arXiv
```
python experiments/run_single_experiment.py --dataset arxiv --root <DATA_ROOT>  -ds_cap 510 -val_cap 510 -test_cap 500 -eval_step 100 -epochs 1 --layers S2,U,M -way 3 -shot 3 -qry 3 -lr 1e-5 -bert roberta-base-nli-stsb-mean-tokens -pretrained <PATH_TO_CHECKPOINT> --eval_only True --train_cap 10 --device 0 
```


<details>
<summary>Commands for Other Configurations and Datasets</summary>
Pretraining for PG-NM and PG-MT. (Evalution code is the same as PRODIGY.)

```
python experiments/run_single_experiment.py --dataset mag240m --root <DATA_ROOT> --original_features True -ds_cap 10010 -val_cap 100 -test_cap 100 --epochs 1 -ckpt_step 1000 -layers S2,U,M -lr 3e-4 -way 30 -shot 3 -qry 4 -eval_step 500 -task neighbor_matching  -bs 1 -aug ND0.5,NZ0.5 -aug_test True -attr 1000 --device 0 --prefix MAG_PG_NM

python experiments/run_single_experiment.py --dataset mag240m --root <DATA_ROOT> --original_features True -ds_cap 10010 -val_cap 100 -test_cap 100 --epochs 1 -ckpt_step 1000 -layers S2,U,M -lr 3e-4 -way 30 -shot 3 -qry 4 -eval_step 500 -task classification  -bs 1 -aug ND0.5,NZ0.5 -aug_test True -attr 1000 --device 0 --prefix MAG_PG_MT
```

Pretraining for Contrastive
```
python experiments/run_single_experiment.py --dataset mag240m --root <DATA_ROOT> --original_features True --input_dim 768 --emb_dim 256 -ds_cap 10010 -val_cap 100 -test_cap 100 --epochs 1 -ckpt_step 1000 -layers S2,U,A -lr 1e-3 -way 30 -shot 1 -qry 4 -eval_step 500 -task same_graph  -bs 1 -aug ND0.5,NZ0.5 -aug_test True --device 0 --prefix MAG_Contrastive
```

Evaluation for Contrastive
```
python experiments/run_single_experiment.py --dataset arxiv --root <DATA_ROOT> --emb_dim 256 --input_dim 768 -ds_cap 510 -val_cap 510 -test_cap 500 -eval_step 100 -epochs 1 --layers S2,U,A -way 3 -shot 3 -qry 3 -lr 1e-5 -bert roberta-base-nli-stsb-mean-tokens  -pretrained <PATH_TO_CHECKPOINT> --eval_only True --train_cap 10 --device 0
```


Execute `kg_commands.py` for examples of pretraining and evaluation commands for KG datasets (uncomment code inside for all commands).


Preprocessing and data loading code for some graph datasets. See `DATASETS.md` for dataset info.
</details>

# Citations
If you use this repo, please cite the following paper. This repo reuses code from [CSR](https://github.com/snap-stanford/csr) for KG datasets loading.
```
@article{Huang2023PRODIGYEI,
  title={PRODIGY: Enabling In-context Learning Over Graphs},
  author={Qian Huang and Hongyu Ren and Peng Chen and Gregor Kr\v{z}manc and Daniel Zeng and Percy Liang and Jure Leskovec},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.12600}
}
```
