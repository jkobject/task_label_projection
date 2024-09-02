import sys
import random
import numpy as np
import anndata as ad
import openproblems

## VIASH START
par = {
    'input': 'resources_test/common/pancreas/dataset.h5ad',
    'method': 'batch',
    'seed': None,
    'obs_batch': 'batch',
    'obs_label': 'cell_type',
    'output_train': 'train.h5ad',
    'output_test': 'test.h5ad',
    'output_solution': 'solution.h5ad'
}
meta = {
    'resources_dir': 'src/tasks/label_projection/process_dataset',
    'config': 'src/tasks/label_projection/process_dataset/.config.vsh.yaml'
}
## VIASH END

# import helper functions
sys.path.append(meta['resources_dir'])
from subset_h5ad_by_format import subset_h5ad_by_format

# set seed if need be
if par["seed"]:
    print(f">> Setting seed to {par['seed']}")
    random.seed(par["seed"])

print(">> Load data", flush=True)
adata = ad.read_h5ad(par["input"])
print("input:", adata)

print(">> Load config", flush=True)
config = openproblems.project.read_viash_config(meta["config"])

print(f">> Process data using {par['method']} method")
if par["method"] == "batch":
    batch_info = adata.obs[par["obs_batch"]]
    batch_categories = batch_info.dtype.categories
    test_batches = random.sample(list(batch_categories), 1)
    is_test = [ x in test_batches for x in batch_info ]
elif par["method"] == "random":
    train_ix = np.random.choice(adata.n_obs, round(adata.n_obs * 0.8), replace=False)
    is_test = [ not x in train_ix for x in range(0, adata.n_obs) ]

# subset the different adatas
print(">> Figuring which data needs to be copied to which output file", flush=True)
# use par arguments to look for label and batch value in different slots
field_rename_dict = {
    "obs": {
        "label": par["obs_label"],
        "batch": par["obs_batch"],
    }
}

print(">> Creating train data", flush=True)
output_train = subset_h5ad_by_format(
    adata[[not x for x in is_test]], 
    config,
    "output_train",
    field_rename_dict
)

print(">> Creating test data", flush=True)
output_test = subset_h5ad_by_format(
    adata[is_test],
    config,
    "output_test",
    field_rename_dict
)

print(">> Creating solution data", flush=True)
output_solution = subset_h5ad_by_format(
    adata[is_test],
    config,
    "output_solution",
    field_rename_dict
)

print(">> Writing data", flush=True)
output_train.write_h5ad(par["output_train"])
output_test.write_h5ad(par["output_test"])
output_solution.write_h5ad(par["output_solution"])
