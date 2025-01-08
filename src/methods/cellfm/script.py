import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import mindspore as ms
import os
import sys
import shutil
import torch
from huggingface_hub import hf_hub_download
from scipy.sparse import csr_matrix 

## VIASH START
par = {
  "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
  "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
  "output": "output.h5ad",
  "model": "path/to/cellfm/model"
}
meta = {
  "name": "cellfm"
}
## VIASH END

print(">>> Reading input files", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

# Code has hardcoded paths that only work correctly inside the CellFM directory
if os.path.isdir("/workspace/CellFM"):
    # For executable we can work inside the CellFM directory
    os.chdir("/workspace/CellFM")
else:
    # For Nextflow we need to copy files to the Nextflow working directory
    print(">>> Copying CellFM files to local directory...", flush=True)
    shutil.copytree("/workspace/CellFM", ".", dirs_exist_ok=True)

# Append current directory to import CellFM functions
sys.path.append(".")

from config import Config
from tutorials.CellAnnotation.annotation_model import *
from metrics import annote_metric
from utils import Wrapper
from data_process import Prepare
from train import freeze_module


class SCrna():
  def __init__(self, adata, mode="train", prep=True):
    if mode == "train":
      adata = adata[adata.obs["train"] == 0]
    elif mode == "val":
      adata = adata[adata.obs["train"] == 1]
    else:
      adata = adata[adata.obs["train"] == 2]
    self.gene_info = pd.read_csv(f"csv/expand_gene_info.csv", index_col=0, header=0)
    self.geneset = {j:i+1 for i, j in enumerate(self.gene_info.index)}
    gene = np.intersect1d(adata.var_names, self.gene_info.index)
    adata = adata[:, gene].copy()
    adata.obs["cell_type"] = adata.obs["label"].astype("category")
    label = adata.obs["cell_type"].cat.codes.values
    adata.obs["label"] = label
    if prep:
      adata.layers["x_normed"] = sc.pp.normalize_total(adata,target_sum=1e4, inplace=False)["X"]
      adata.layers["x_log1p"] = adata.layers["x_normed"]
      sc.pp.log1p(adata, layer="x_log1p")
    self.adata = adata
    self.id2label = adata.obs["cell_type"].cat.categories.values
    self.gene = np.array([self.geneset[i] for i in self.adata.var_names]).astype(np.int32)
    self.cls = len(adata.obs["cell_type"].unique())
    self.label = self.adata.obs["label"].values.astype(np.int32)
    print(f"{mode} adata:", adata.shape, self.cls)
    if prep:
      self.data = self.adata.layers["x_log1p"].A.astype(np.float32)
    else:
      self.data = self.adata.X.astype(np.int32)
  
  def __len__(self):
    return len(self.adata)
  
  def __getitem__(self,idx):
    data = self.data[idx].reshape(-1)
    label = self.label[idx]
    return data, self.gene, label
    

# Creating a data loader
def build_dataset(
  data, prep, batch,
  rank_size = None,
  rank_id = None,
  drop = True,
  shuffle = True
):
  dataset = ms.dataset.GeneratorDataset(
    data, 
    column_names =["data", "gene", "label"],
    shuffle=shuffle,
    num_shards=rank_size, 
    shard_id=rank_id
  )
  dataset = dataset.map(
    prep.seperate, 
    input_columns=["data"],
    output_columns=["data", "nonz", "zero"]
  )
  dataset = dataset.map(
    prep.sample, 
    input_columns=["data", "nonz", "zero"],
    output_columns=["data", "nonz", "cuted", "z_sample", "seq_len"]
  )
  dataset = dataset.map(
    prep.compress, 
    input_columns=["data", "nonz"],
    output_columns=["data", "nonz_data", "nonz"]
  )
  dataset = dataset.map(
    prep.compress, 
    input_columns=["gene","nonz"],
    output_columns=["gene", "nonz_gene", "nonz"]
  )
  dataset = dataset.map(
    prep.attn_mask, 
    input_columns=["seq_len"],
    output_columns=["zero_idx"]
  )
  dataset = dataset.map(
    prep.pad_zero, 
    input_columns=["nonz_data"]
  )
  dataset = dataset.map(
    prep.pad_zero, 
    input_columns=["nonz_gene"]
  )
  dataset = dataset.project(
    columns=["nonz_data", "nonz_gene", "zero_idx", "label"]
  )
  dataset = dataset.batch(
    batch,
    num_parallel_workers=4, 
    drop_remainder=drop, 
  )
  return dataset


# Choose the type and number of the GPU
if torch.cuda.is_available():
  ms.set_context(
    device_target='GPU', 
    mode=ms.GRAPH_MODE,
    device_id=0,
  )
ms.set_seed(0)

print(">>> Preprocess data", flush=True)
input_train.var_names = input_train.var["feature_name"]
input_test.var_names = input_test.var["feature_name"]
input_train.obs['train'] = 0
input_test.obs['train']  = 2

adata = ad.concat([input_train, input_test], join='outer')
print('origin shape:', adata.shape, len(adata.obs['label'].unique()))

data = adata.layers['counts'].astype(np.float32)
T = adata.layers['counts'].sum(1)
data = csr_matrix(np.round(data/np.maximum(1, T/1e5, dtype=np.float32)))
data.eliminate_zeros()
adata.X = data

trainset = SCrna(adata, mode="train")
testset = SCrna(adata, mode="test")

cfg = Config()
cfg.num_cls = trainset.cls
cfg.enc_nlayers = 2
if len(trainset.gene) < cfg.nonz_len: 
  cfg.nonz_len = len(trainset.gene)

prep = Prepare(cfg.nonz_len, pad=1, mask_ratio=0, random=False)
train_loader = build_dataset(
  trainset,
  prep,
  16,
  drop=True,
  shuffle=True,
)
test_loader = build_dataset(
  testset,
  prep,
  1,
  drop=False,
  shuffle=False,
)

print(">>> Train model", flush=True)

# Create the training model for CellFM and freeze the parameters of its backbone layer

model_checkpoint = hf_hub_download(repo_id="ShangguanNingyuan/CellFM", filename=f"CellFM_80M_weight.ckpt")
para = ms.load_checkpoint(model_checkpoint)

backbone = Backbone(len(trainset.geneset), cfg)
ms.load_param_into_net(backbone, para)
model = Net(backbone, cfg)

freeze_module(model.extractor)

optimizer = ms.nn.Adam(model.trainable_params(), 1e-4, weight_decay=1e-5)
update_cell = ms.nn.DynamicLossScaleUpdateCell(1, 2, 1000)
wrapper = Wrapper(model, optimizer)
trainer = ms.train.Model(
  wrapper,
  eval_network=model,
  amp_level='O0',
  metrics={'accuracy': annote_metric(trainset.cls, key='accuracy')},
  eval_indexes=[0, 1, 2]
)
loss_cb = ms.train.LossMonitor(20)

ckpt_config = ms.train.CheckpointConfig(
  save_checkpoint_steps=len(train_loader),
  keep_checkpoint_max=1,
  integrated_save=False,
  async_save=False
)
ckpt_cb = ms.train.ModelCheckpoint(
  prefix=f'zeroshot', 
  config=ckpt_config
)
cbs = [loss_cb, ckpt_cb]

# Model training 
trainer.train(30, train_loader, callbacks=cbs)


print(">>> Generate predictions", flush=True)
# Model evaluation
ms.load_param_into_net(model, ms.load_checkpoint(ckpt_cb.latest_ckpt_file_name))
trainer.eval(test_loader)
# extract predictions

print(">>> Write output AnnData to file", flush=True)
output = ad.AnnData(
  obs=input_test.obs[["label_pred"]],
  uns={
    'method_id': meta['name'],
    'dataset_id': input_test.uns['dataset_id'],
    'normalization_id': input_test.uns['normalization_id']
  }

)
output.write_h5ad(par["output"], compression="gzip")