# -*- coding: utf-8 -*-
"""单折训练入口：python -m src.train.run_fold --subtask B --fold 环保 --seed 42

流程：加载配置 -> 准备数据（若无则先 prepare/split）-> 建模型/数据 -> 训练 -> 评估 -> 存 ckpt。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.common.config import Config
from src.data.dataset import StanceDataset, collate_fn
from src.data.split import load_split
from src.models import StanceClassifier
from src.train.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subtask", default="B", choices=["A", "B"])
    p.add_argument("--fold", default=None, help="Subtask B 的留出域（如 环保）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.load()

    if not (cfg.path("interim") / "all.parquet").exists():
        from src.data.prepare import run as run_prepare
        from src.data.split import run as run_split
        run_prepare(cfg)
        run_split(cfg)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, cache_dir=cfg.base["model"]["cache_dir"])
    dg = cfg.dg
    num_domains = dg["domain"]["num_real_domains"] + len(dg["domain"]["aug_styles"]) \
        if dg["loss"].get("domain", 0) > 0 else 0
    model = StanceClassifier(cfg.backbone, cfg.num_labels,
                             dropout=cfg.base["model"]["dropout"],
                             cache_dir=cfg.base["model"]["cache_dir"],
                             num_domains=num_domains,
                             disc_hidden=dg["domain"]["discriminator_hidden"])

    def make_loader(split):
        df = load_split(cfg, args.subtask, split, fold=args.fold)
        ds = StanceDataset(df, tokenizer, cfg.input_template, cfg.max_length)
        return DataLoader(ds, batch_size=cfg.base["data"]["batch_size"],
                          shuffle=(split == "train"), collate_fn=collate_fn,
                          num_workers=cfg.base["data"]["num_workers"])

    trainer = Trainer(model, dg["loss"], dg, cfg.base["training"],
                      device=device, log_dir=cfg.path("outputs") / "logs")
    f1 = trainer.train(make_loader("train"), make_loader("val"),
                       epochs=cfg.base["training"]["epochs"])

    ckpt = cfg.path("outputs") / "ckpt" / f"{args.subtask}_{args.fold or 'all'}_seed{args.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "f1": f1, "seed": args.seed}, ckpt)
    print(f"[done] subtask={args.subtask} fold={args.fold or '-'} seed={args.seed} "
          f"val_macro_f1={f1:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
