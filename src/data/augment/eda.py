# -*- coding: utf-8 -*-
"""EDA 简易文本增强（Wei & Zou, EMNLP-IJCNLP 2019）。

中文版注意事项：同义词替换必须保护极性词（支持/反对的立场极性词不可随意替换），
否则会翻转立场。当前实现提供随机插入/交换/删除的结构性操作；
同义词替换需外部同义词表（TODO），默认关闭。
"""
from __future__ import annotations

import random

import pandas as pd

from .base import Augmenter


class EDA(Augmenter):
    def __init__(self, cfg: dict, synonym_dict: dict | None = None):
        super().__init__(cfg)
        self.alpha_sr = cfg.get("alpha_sr", 0.1)
        self.p_ri = cfg.get("p_ri", 0.1)
        self.p_rs = cfg.get("p_rs", 0.1)
        self.p_rd = cfg.get("p_rd", 0.1)
        self.synonym_dict = synonym_dict or {}  # {词: [同义词...]}，需含极性词保护

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            text = self._random_insert(r["text"])
            text = self._random_swap(text)
            text = self._random_delete(text)
            if self.synonym_dict and random.random() < self.alpha_sr:
                text = self._synonym_replace(text)
            if text != r["text"]:
                row = r.to_dict()
                row["text"] = text
                row["aug_style"] = "eda"
                rows.append(row)
        return pd.DataFrame(rows)

    # ---- 结构性操作（字符级近似，中文按字粒度处理）----
    def _random_insert(self, text: str) -> str:
        if random.random() > self.p_ri or not text:
            return text
        chars = list(text)
        pos = random.randint(0, len(chars) - 1)
        chars.insert(pos, chars[pos])  # 简单复制邻近字（更精细需近义词插入）
        return "".join(chars)

    def _random_swap(self, text: str) -> str:
        if random.random() > self.p_rs or len(text) < 2:
            return text
        chars = list(text)
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def _random_delete(self, text: str) -> str:
        if random.random() > self.p_rd or not text:
            return text
        chars = list(text)
        del chars[random.randint(0, len(chars) - 1)]
        return "".join(chars)

    def _synonym_replace(self, text: str) -> str:
        # TODO: 用外部同义词表替换，且跳过立场极性词（保护）
        return text
