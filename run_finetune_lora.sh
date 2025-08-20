#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh   # ← 新增
conda activate openvla                       # ← 新增

export DATA_ROOT=${DATA_ROOT:-$HOME/my_projects/datasets}
export RUN_ROOT=${RUN_ROOT:-$HOME/my_projects/runs/openvla}
export ADAPTER_TMP=${ADAPTER_TMP:-$HOME/my_projects/tmp/openvla-adapter}
export WANDB_DISABLED=true
mkdir -p "$RUN_ROOT" "$ADAPTER_TMP"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name bridge_orig \
  --run_root_dir "$RUN_ROOT" \
  --adapter_tmp_dir "$ADAPTER_TMP" \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True
