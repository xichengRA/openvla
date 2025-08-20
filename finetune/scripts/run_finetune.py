#!/usr/bin/env python
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML/JSON config")
    p.add_argument("--output_dir", default="finetune/checkpoints")
    args = p.parse_args()
    print(f"[finetune] Using config: {args.config}")
    print(f"[finetune] Outputs will be saved to: {args.output_dir}")
    # TODO: load config, build model/dataset, start training

if __name__ == "__main__":
    main()
