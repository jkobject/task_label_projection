#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

DATASETS_DIR="resources_test/task_label_projection"
OUTPUT_DIR="output/temp"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export NXF_VER=24.04.3
nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  -entry auto \
  -c common/nextflow_helpers/labels_ci.config \
  --input_states "$DATASETS_DIR/cxg_immune_cell_atlas/state.yaml" \
  --rename_keys 'input_train:output_train;input_test:output_test;input_solution:output_solution' \
  --settings '{"output_scores": "scores.yaml", "output_dataset_info": "dataset_info.yaml", "output_method_configs": "method_configs.yaml", "output_metric_configs": "metric_configs.yaml", "output_task_info": "task_info.yaml"}' \
  --publish_dir "$OUTPUT_DIR" \
  --output_state "state.yaml"
