# -*- coding: utf-8 -*-
"""回译增强（中->英->中，Sennrich et al., ACL 2016）。

增加表层多样性；风险：回译可能翻转立场极性，需经标签一致性闸门校验。
依赖 transformers MarianMT（本地 CPU 也可运行，较慢）。
"""
from __future__ import annotations

import pandas as pd

from .base import Augmenter


class BackTranslation(Augmenter):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model_name = cfg.get("model", "Helsinki-NLP/opus-mt-zh-en")
        self.back_model = cfg.get("back_model", "Helsinki-NLP/opus-mt-en-zh")
        self._fwd = self._bwd = None  # 惰性加载

    def _load(self):
        if self._fwd is None:
            from transformers import MarianMTModel, MarianTokenizer
            self._fwd = (MarianTokenizer.from_pretrained(self.model_name),
                         MarianMTModel.from_pretrained(self.model_name))
            self._bwd = (MarianTokenizer.from_pretrained(self.back_model),
                         MarianMTModel.from_pretrained(self.back_model))
        return self._fwd, self._bwd

    def _translate(self, tokenizer, model, text: str) -> str:
        import torch
        enc = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model.generate(**enc)
        return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            try:
                fwd, bwd = self._load()
                en = self._translate(fwd[0], fwd[1], r["text"])
                back = self._translate(bwd[0], bwd[1], en)
                if back.strip() and back != r["text"]:
                    row = r.to_dict()
                    row["text"] = back
                    row["aug_style"] = "backtranslation"
                    rows.append(row)
            except Exception as e:  # 单条失败不影响整体
                print(f"  [backtranslation] 跳过一条：{e}")
        return pd.DataFrame(rows)
