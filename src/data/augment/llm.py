# -*- coding: utf-8 -*-
"""LLM 条件生成增强（目标改写 / 文本改写）。

直击「94% 目标只出现一次」的目标稀疏问题：让 LLM 生成同义目标改写，
或保持立场极性的文本改写。生成时须屏蔽测试目标，防止目标泄露。
"""
from __future__ import annotations

import pandas as pd

from .base import Augmenter


class LLMAugmenter(Augmenter):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = cfg.get("model")  # 本地模型 id 或 API 客户端
        self.style = cfg.get("style", "rewrite_target")

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: 接入具体 LLM（Qwen/GLM 本地推理，或 OpenAI 兼容 API）。
        # 提示词需明确：保持立场极性；不要提及测试目标；输出 JSON 便于解析。
        raise NotImplementedError(
            "LLMAugmenter 需接入具体 LLM 后端；见 configs/da.yaml 的 methods.llm.model")
