#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, subprocess, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[run_lora] Please install pyyaml: pip install pyyaml")
    sys.exit(1)

def as_str(v):
    return "True" if v is True else "False" if v is False else str(v)

def main():
    p = argparse.ArgumentParser(description="LoRA finetune launcher for OpenVLA")
    p.add_argument("--config", required=True, help="Path to YAML config")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # vla_path 可填 HF 模型ID（如 "openvla/openvla-7b"）或本地模型目录
    vla_path = cfg.get("vla_path", "openvla/openvla-7b")
    nnodes = int(cfg.get("nnodes", 1))
    nproc  = int(cfg.get("nproc_per_node", 1))

    data_root_dir  = cfg["data_root_dir"]
    dataset_name   = cfg.get("dataset_name", "bridge_orig")
    run_root_dir   = cfg["run_root_dir"]
    adapter_tmp_dir= cfg.get("adapter_tmp_dir", os.path.join(run_root_dir, "adapters"))

    lora_rank   = int(cfg.get("lora_rank", 32))
    batch_size  = int(cfg.get("batch_size", 8))
    grad_accum  = int(cfg.get("grad_accumulation_steps", 1))
    lr          = cfg.get("learning_rate", 5e-4)
    image_aug   = cfg.get("image_aug", True)
    save_steps  = int(cfg.get("save_steps", 500))

    wandb_project = cfg.get("wandb_project")
    wandb_entity  = cfg.get("wandb_entity")

    # 确保输出目录存在
    for d in [run_root_dir, adapter_tmp_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun", "--standalone",
        "--nnodes", str(nnodes),
        "--nproc-per-node", str(nproc),
        "vla-scripts/finetune.py",
        "--vla_path", vla_path,
        "--data_root_dir", data_root_dir,
        "--dataset_name", dataset_name,
        "--run_root_dir", run_root_dir,
        "--adapter_tmp_dir", adapter_tmp_dir,
        "--lora_rank", str(lora_rank),
        "--batch_size", str(batch_size),
        "--grad_accumulation_steps", str(grad_accum),
        "--learning_rate", as_str(lr),
        "--image_aug", as_str(image_aug),
        "--save_steps", str(save_steps),
    ]
    if wandb_project: cmd += ["--wandb_project", wandb_project]
    if wandb_entity:  cmd += ["--wandb_entity",  wandb_entity]

    print("\n[run_lora] Launch command:\n", " ".join(cmd), "\n", flush=True)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
