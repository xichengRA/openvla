# Finetune

LoRA fine-tuning entry lives here.

## Structure
- `scripts/`   : launchers
- `configs/`   : example configs
- `data/`, `logs/`, `checkpoints/` : local-only; ignored by git

## LoRA wrapper

`finetune/scripts/run_lora.py` reads a YAML config and launches
`vla-scripts/finetune.py` via `torchrun`.

- `vla_path` accepts **HF model id** (e.g. `openvla/openvla-7b`) **or** a **local model dir**.

### Local machine (cloud) config (untracked)
Create your own config under `finetune/configs/local/` (ignored by git), e.g.:

```yaml
vla_path: "openvla/openvla-7b"
nnodes: 1
nproc_per_node: 1
data_root_dir: "/home/ubuntu/my_projects/datasets"
dataset_name: "bridge_orig"
run_root_dir: "/home/ubuntu/my_projects/runs/bridge_lora"
adapter_tmp_dir: "/home/ubuntu/my_projects/runs/bridge_lora/adapters"
lora_rank: 32
batch_size: 8
grad_accumulation_steps: 2
learning_rate: 5e-4
image_aug: true
save_steps: 500
