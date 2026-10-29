# -*- coding: utf-8 -*-
"""单折训练入口（当前研究只使用 Subtask B 领域级零样本）。

用法：
  python -m src.train.run_fold --subtask B --fold 环保 --seed 42
  python -m src.train.run_fold --subtask B --fold 环保 --seed 42 --loss-config dg

流程：配置 -> 数据 -> 模型 -> 训练（warmup + AMP）-> test 评测 -> 保存 ckpt / predictions / metrics。
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.common.config import Config, PROJECT_ROOT
from src.data.constants import DOMAIN_ORDER
from src.data.dataset import StanceDataset, collate_fn
from src.data.split import load_split
from src.eval.metrics import full_report, neutral_f1
from src.models import StanceClassifier
from src.train.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subtask", default="B", choices=["A", "B"],
                   help="当前实验只使用 B；A 保留兼容")
    p.add_argument("--fold", default=None, help="Subtask B 的留出域（如 环保）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss-config", default="baseline", choices=["baseline", "dg"],
                   help="baseline=CE-only 基线；dg=SupCon+Proto+DANN 完整损失")
    p.add_argument("--epochs", type=int, default=None,
                   help="覆盖 configs/base.yaml 的 training.epochs（冒烟测试用，默认读配置）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="覆盖 configs/base.yaml 的 data.batch_size（显存受限时用，默认读配置）")
    return p.parse_args()


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 的随机种子（torch.initial_seed 已由主进程 seed 派生）。"""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    if args.subtask == "B":
        if not args.fold:
            raise SystemExit("Subtask B 必须指定 --fold（例如 --fold 环保）")
        if args.fold not in DOMAIN_ORDER:
            raise SystemExit(f"未知领域 {args.fold!r}；可选：{DOMAIN_ORDER}")

    cfg = Config.load(dg=f"{args.loss_config}.yaml")
    dg = cfg.dg
    # 复制后按 CLI 覆盖，避免改动 cfg.base 里的原始配置
    training_cfg = dict(cfg.base["training"])
    data_cfg = dict(cfg.base["data"])
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size

    # ---- 数据：interim 缺失时重建；splits 缺失（或指定 fold 缺失）时重建 ----
    if not (cfg.path("interim") / "all.parquet").exists():
        from src.data.prepare import run as run_prepare
        from src.data.split import run as run_split
        run_prepare(cfg)
        run_split(cfg)
    else:
        # split.py 把 manifest 写在 data/processed/split_manifest.json（splits 的上级目录）
        manifest = cfg.path("splits").parent / "split_manifest.json"
        need_split = not manifest.exists()
        if args.subtask == "B" and not (cfg.path("splits") / "B" / args.fold / "train.parquet").exists():
            need_split = True
        if need_split:
            from src.data.split import run as run_split
            run_split(cfg)

    deterministic = bool(training_cfg.get("deterministic", True))
    set_seed(args.seed, deterministic=deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.backbone, cache_dir=cfg.base["model"].get("cache_dir"))

    # Subtask A 没有领域标签，DANN 无法训练；B 才启用域判别器。
    use_domain = args.subtask == "B" and float(dg["loss"].get("domain", 0.0)) > 0
    if args.subtask == "A" and float(dg["loss"].get("domain", 0.0)) > 0:
        print("[warn] Subtask A 的 domain_id 为 -1，自动关闭 DANN（其余损失照常）")
    num_domains = (dg["domain"]["num_real_domains"] + len(dg["domain"]["aug_styles"])) \
        if use_domain else 0

    model = StanceClassifier(cfg.backbone, cfg.num_labels,
                             dropout=cfg.base["model"]["dropout"],
                             cache_dir=cfg.base["model"].get("cache_dir"),
                             num_domains=num_domains,
                             disc_hidden=dg["domain"]["discriminator_hidden"])

    def make_loader(split: str) -> DataLoader:
        df = load_split(cfg, args.subtask, split, fold=args.fold)
        ds = StanceDataset(df, tokenizer, cfg.input_template, cfg.max_length)
        generator = torch.Generator()
        generator.manual_seed(args.seed + {"train": 0, "val": 1, "test": 2}[split])
        return DataLoader(
            ds,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=(device == "cuda"),
            worker_init_fn=seed_worker,
            generator=generator,
        )

    train_loader = make_loader("train")
    val_loader = make_loader("val")
    test_loader = make_loader("test")

    trainer = Trainer(model, dg["loss"], dg, training_cfg,
                      device=device, log_dir=cfg.path("outputs") / "logs")

    epochs = int(training_cfg["epochs"])
    total_steps = max(1, len(train_loader) * epochs)
    warmup_ratio = min(max(float(training_cfg.get("warmup_ratio", 0.1)), 0.0), 1.0)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        trainer.optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    print(f"[run] subtask={args.subtask} fold={args.fold or '-'} seed={args.seed} "
          f"loss={args.loss_config} device={device} amp={trainer.use_amp} "
          f"max_length={cfg.max_length} batch_size={data_cfg['batch_size']}")
    print(f"[steps] epochs={epochs} steps/epoch={len(train_loader)} "
          f"total={total_steps} warmup={warmup_steps}")

    val_f1 = trainer.train(train_loader, val_loader, epochs=epochs,
                           scheduler=scheduler, total_steps=total_steps)

    # ---- test 评测（checkpoint 已由 Trainer 恢复为 val 最优）----
    y_true, y_pred, probs = trainer.predict(test_loader)
    report = full_report(y_true, y_pred, cfg.id_to_label)
    neutral_id = cfg.label_to_id["中立"]
    test_neutral_f1 = neutral_f1(y_true, y_pred, neutral_id=neutral_id)

    out_root = cfg.path("outputs")
    ckpt_dir = out_root / "ckpt"
    pred_dir = out_root / "predictions"
    table_dir = out_root / "tables"
    for d in (ckpt_dir, pred_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)
    stem = f"{args.subtask}_{args.fold or 'all'}_seed{args.seed}"
    ckpt_path = ckpt_dir / f"{stem}.pt"
    metrics_path = table_dir / f"{stem}_metrics.json"
    pred_path = pred_dir / f"{stem}_predictions.parquet"

    def rel(p) -> str:
        return p.relative_to(PROJECT_ROOT).as_posix()

    metrics = {
        "subtask": args.subtask,
        "fold": args.fold,
        "seed": args.seed,
        "loss_config": args.loss_config,
        "backbone": cfg.backbone,
        "max_length": cfg.max_length,
        "batch_size": int(data_cfg["batch_size"]),
        "epochs": epochs,
        "learning_rate": float(training_cfg["learning_rate"]),
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "use_amp": trainer.use_amp,
        "deterministic": deterministic,
        "val_macro_f1": float(val_f1),
        "test_macro_f1": float(report["macro_f1"]),
        "test_neutral_f1": float(test_neutral_f1),
        "per_class_f1": report["per_class_f1"],
        "confusion_matrix": report["confusion_matrix"],
        "classification_report": report["classification_report"],
        "n_test": int(len(y_true)),
        "labels": cfg.labels,
        "loss_weights": {k: float(v) for k, v in dg["loss"].items()},
        "training_history": trainer.history,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {"ckpt": rel(ckpt_path), "predictions": rel(pred_path)},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---- 逐样本预测（顺序与 test split 一致）----
    test_df = test_loader.dataset.df.reset_index(drop=True)
    if len(test_df) != len(y_pred):
        raise RuntimeError(f"预测行数 {len(y_pred)} 与 test 行数 {len(test_df)} 不一致")
    keep = [c for c in ("row_id", "text", "target", "stance_id", "target_type", "domain")
            if c in test_df.columns]
    pred_df = test_df[keep].copy()
    pred_df["pred_id"] = y_pred.astype(int)
    pred_df["pred_label"] = pred_df["pred_id"].map(cfg.id_to_label)
    pred_df["correct"] = pred_df["pred_id"] == pred_df["stance_id"]
    for i, _label in enumerate(cfg.labels):
        pred_df[f"prob_{i}"] = probs[:, i].astype(float)
    pred_df.to_parquet(pred_path, index=False)

    torch.save({
        "state_dict": model.state_dict(),
        "f1": float(val_f1),                 # 兼容旧字段：val macro-F1
        "val_macro_f1": float(val_f1),
        "test_macro_f1": float(report["macro_f1"]),
        "test_neutral_f1": float(test_neutral_f1),
        "seed": args.seed,
        "subtask": args.subtask,
        "fold": args.fold,
        "loss_config": args.loss_config,
        "backbone": cfg.backbone,
        "max_length": cfg.max_length,
        "labels": cfg.labels,
    }, ckpt_path)

    print(f"[done] subtask={args.subtask} fold={args.fold or '-'} seed={args.seed} "
          f"val_macro_f1={val_f1:.4f} test_macro_f1={report['macro_f1']:.4f} "
          f"test_neutral_f1={test_neutral_f1:.4f}")
    print(f"  ckpt      -> {ckpt_path}")
    print(f"  metrics   -> {metrics_path}")
    print(f"  predictions -> {pred_path}")


if __name__ == "__main__":
    main()
