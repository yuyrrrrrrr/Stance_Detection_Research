# -*- coding: utf-8 -*-
"""PyTorch Dataset 与 collate：加载 parquet，经 build_input 统一模板化后分词。"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.common.format import build_input


class StanceDataset(Dataset):
    """一条样本 -> {input_ids, attention_mask, label, domain_id, target_id}。

    stance_id（立场标签 0/1/2）读入后放入 batch key "label"；target_id 供 SupCon 区分
    「同/不同目标」；domain_id 供 DANN 域判别。
    """

    def __init__(self, df, tokenizer, template: str, max_length: int,
                 stance_id_col: str = "stance_id", domain_id_col: str = "domain_id",
                 target_id_col: str = "target_id"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.stance_id_col = stance_id_col
        self.domain_id_col = domain_id_col
        self.target_id_col = target_id_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        inp = build_input(row["text"], row["target"], self.template, self.max_length)
        enc = self.tokenizer(inp, truncation=True, max_length=self.max_length,
                             add_special_tokens=True, return_tensors=None)
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": int(row[self.stance_id_col]),  # stance_id(0/1/2) -> batch key "label"
        }
        if self.domain_id_col in self.df.columns:
            item["domain_id"] = int(row[self.domain_id_col])
        if self.target_id_col in self.df.columns:
            item["target_id"] = int(row[self.target_id_col])
        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """动态 padding，并把 label/domain_id/target_id 堆成张量。"""
    from transformers import DataCollatorWithPadding  # 延迟导入避免硬依赖

    # 简化实现：自建 padding（避免依赖 DataCollatorWithPadding 的 tokenizer 引用）
    pad_id = 0  # BERT/RoBERTa 的 pad_token_id
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attn = [], []
    for b in batch:
        n = len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * (max_len - n))
        attn.append(b["attention_mask"] + [0] * (max_len - n))

    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }
    if "domain_id" in batch[0]:
        out["domain_id"] = torch.tensor([b["domain_id"] for b in batch], dtype=torch.long)
    if "target_id" in batch[0]:
        out["target_id"] = torch.tensor([b["target_id"] for b in batch], dtype=torch.long)
    return out
