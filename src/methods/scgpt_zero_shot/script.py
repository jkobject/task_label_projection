import anndata as ad
import gdown
import scgpt
import torch
import tempfile
import numpy as np
import pandas as pd

## VIASH START
par = {
  'input_train': 'resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad',
  'input_test': 'resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad',
  'output': 'output.h5ad'
}
meta = {
  'name': 'scgpt'
}
## VIASH END

def l2_sim(a, b):
  sims = -np.linalg.norm(a - b, axis=1)
  return sims

def get_similar_vectors(vector, ref, top_k=10):
  sims = l2_sim(vector, ref)
  top_k_idx = np.argsort(sims)[::-1][:top_k]
  return top_k_idx, sims[top_k_idx]

print('Reading input files', flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])

if input_train.uns["dataset_organism"] != "homo_sapiens":
  raise ValueError(
    f"scGPT can only be used with human data "
    f"(dataset_organism is \"{input_train.uns['dataset_organism']}\")"
  )

drive_path = f"https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
print(f'Downloading scGPT_human model from {drive_path}', flush=True)
model_dir = tempfile.TemporaryDirectory()
gdown.download_folder(drive_path, output=model_dir.name, quiet=True)
print(f"Model directory: '{model_dir.name}'", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: '{device}'", flush=True)

print('Preprocessing and embedding train data', flush=True)
input_train.X = input_train.layers["counts"]
ref_embed = scgpt.tasks.embed_data(
  input_train,
  model_dir.name,
  gene_col="feature_name",
  obs_to_save="label",
  batch_size=64,
  device=device,
  use_fast_transformer=False,
  return_new_adata=True,
)

print('Preprocessing and embedding test data', flush=True)
input_test.X = input_test.layers["counts"]
test_embed = scgpt.tasks.embed_data(
  input_test,
  model_dir.name,
  gene_col="feature_name",
  batch_size=64,
  device=device,
  use_fast_transformer=False,
  return_new_adata=True,
)

print('Generate predictions', flush=True)
k = 10  # number of neighbors
idx_list=[i for i in range(test_embed.X.shape[0])]
preds = []
for k in idx_list:
  idx, sim = get_similar_vectors(test_embed.X[k][np.newaxis, ...], ref_embed.X, k)
  pred = ref_embed.obs["label"][idx].value_counts()
  preds.append(pred.index[0])

input_test.obs["label_pred"] = preds
input_test.obs["label_pred"] = input_test.obs["label_pred"].astype("category")

print("Write output AnnData to file", flush=True)
output = ad.AnnData(
  obs=input_test.obs[["label_pred"]],
  uns={
    'method_id': meta['name'],
    'dataset_id': input_test.uns['dataset_id'],
    'normalization_id': input_test.uns['normalization_id']
  }
)
output.write_h5ad(par['output'], compression='gzip')

