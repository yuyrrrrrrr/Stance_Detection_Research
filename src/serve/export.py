# -*- coding: utf-8 -*-
"""模型导出：TorchScript / ONNX（用于部署）。

注意：导出前需保证输入模板与训练一致（build_input）。
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.common.config import Config
from src.models import StanceClassifier


def export_torchscript(ckpt_path: str, out_path: str):
    cfg = Config.load()
    model = StanceClassifier(cfg.backbone, cfg.num_labels)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.eval()
    # 以 (batch, seq_len) 的 input_ids + attention_mask 为输入做 trace
    example = torch.zeros(1, cfg.max_length, dtype=torch.long)
    mask = torch.ones(1, cfg.max_length, dtype=torch.long)
    traced = torch.jit.trace(model, (example, mask))
    traced.save(out_path)
    print(f"[export] TorchScript 已保存到 {out_path}")


def export_onnx(ckpt_path: str, out_path: str):
    cfg = Config.load()
    model = StanceClassifier(cfg.backbone, cfg.num_labels)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.eval()
    example = torch.zeros(1, cfg.max_length, dtype=torch.long)
    mask = torch.ones(1, cfg.max_length, dtype=torch.long)
    torch.onnx.export(model, (example, mask), out_path,
                      input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], dynamic_axes={
                          "input_ids": {0: "batch"}, "attention_mask": {0: "batch"}})
    print(f"[export] ONNX 已保存到 {out_path}")
