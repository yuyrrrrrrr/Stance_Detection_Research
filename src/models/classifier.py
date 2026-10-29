# -*- coding: utf-8 -*-
"""立场分类器：预训练底座 + 分类头（可选域判别头）。

底座：hfl/chinese-roberta-wwm-ext（主）/ bert-base-chinese（对照）。
forward 返回 logits；return_feats=True 时额外返回池化特征（供 SupCon/Proto/DANN）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .dann import DomainDiscriminator, GradientReversalLayer


class StanceClassifier(nn.Module):
    def __init__(self, backbone: str, num_labels: int, dropout: float = 0.1,
                 cache_dir: str | None = None, num_domains: int = 0,
                 disc_hidden: int = 256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone, cache_dir=cache_dir)
        self.encoder = AutoModel.from_pretrained(backbone, config=self.config,
                                                 cache_dir=cache_dir)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

        self.num_domains = num_domains
        if num_domains > 0:
            self.grl = GradientReversalLayer()
            self.domain_disc = DomainDiscriminator(hidden, num_domains, disc_hidden, dropout)
        else:
            self.grl = self.domain_disc = None

    def _pool(self, outputs):
        # BERT/RoBERTa 有 pooler_output；否则取 [CLS]
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, return_feats: bool = False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs)
        feats = self.dropout(pooled)
        logits = self.classifier(feats)
        if return_feats:
            return logits, feats
        return logits

    def forward_domain(self, feats):
        """经梯度反转层后送域判别器。"""
        if self.domain_disc is None:
            raise RuntimeError("模型未启用域判别器（num_domains=0）")
        return self.domain_disc(self.grl(feats))
