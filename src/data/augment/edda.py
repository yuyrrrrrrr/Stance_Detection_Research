# -*- coding: utf-8 -*-
"""EDDA（表征级/对抗式数据增强）。

在模型表征空间施加对抗扰动，产生「更难」的增强，与文本级增强互补。
通常作为训练循环内联操作，而非离线生成样本。
"""
from __future__ import annotations

import pandas as pd

from .base import Augmenter


class EDDA(Augmenter):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.epsilon = cfg.get("epsilon", 1.0)
        self.steps = cfg.get("steps", 3)

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: EDDA 依赖模型梯度，需在训练器内联实现（见 src/train/trainer.py 的 TODO）。
        raise NotImplementedError("EDDA 为训练时内联增强，请在 trainer 中实现")
