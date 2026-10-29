# -*- coding: utf-8 -*-
"""损失函数集合。

总损失：L_total = λc·L_CE + λs·L_SupCon + λp·L_Proto + λd·L_Domain
- L_CE      交叉熵（立场分类）
- L_SupCon  监督对比损失，正样本对 = 同立场 + 不同目标（Khosla et al., NeurIPS 2020）
- L_Proto   原型对齐损失（Prototypical Networks, NeurIPS 2017）
- L_Domain  域判别交叉熵（配 DANN 梯度反转）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # 统一在 fp32 计算损失，避免 AMP autocast 下的数值问题
    return F.cross_entropy(logits.float(), labels)


def domain_loss(domain_logits: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(domain_logits.float(), domain_ids)


def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                target_ids: torch.Tensor | None = None,
                temperature: float = 0.1) -> torch.Tensor:
    """监督对比损失（SupCon）。

    正样本对：同 label 且（若给 target_ids）不同 target。
    features 会被 L2 归一化后计算余弦相似度。
    """
    # AMP 下 features 可能是 fp16；exp(sim/τ) 在 fp16 下很容易溢出（τ=0.1 时 exp(10)≈2.2e4），
    # 因此对比/原型损失统一转到 fp32 计算。
    feats = F.normalize(features.float(), dim=1)
    sim = torch.matmul(feats, feats.T) / temperature  # (N, N)

    labels = labels.contiguous().view(-1, 1)
    pos_mask = labels.eq(labels.T)  # 同立场
    if target_ids is not None:
        target_ids = target_ids.contiguous().view(-1, 1)
        pos_mask = pos_mask & ~target_ids.eq(target_ids.T)  # 排除同目标
    pos_mask = pos_mask.fill_diagonal_(False)

    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    exp_logits = logits.exp()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    n_pos = pos_mask.sum(dim=1).clamp(min=1e-9)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / n_pos
    return -mean_log_prob_pos.mean()


def compute_prototypes(features: torch.Tensor, labels: torch.Tensor,
                       num_classes: int) -> torch.Tensor:
    """按类求原型（均值向量）。"""
    feats = features.float()
    protos = torch.zeros(num_classes, feats.size(1), device=feats.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            protos[c] = feats[mask].mean(dim=0)
    return F.normalize(protos, dim=1)


def proto_loss(features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor,
               temperature: float = 0.1) -> torch.Tensor:
    """原型对齐损失：以样本到各原型的余弦相似度做软标签交叉熵。"""
    feats = F.normalize(features.float(), dim=1)
    logits = torch.matmul(feats, prototypes.float().T) / temperature  # (N, num_classes)
    return F.cross_entropy(logits, labels)


class CombinedLoss(nn.Module):
    """按权重聚合各损失（权重来自 configs/dg.yaml）。"""

    def __init__(self, weights: dict, supcon_t: float = 0.1, proto_t: float = 0.1):
        super().__init__()
        self.weights = weights
        self.supcon_t = supcon_t
        self.proto_t = proto_t

    def forward(self, logits, features, labels,
                target_ids=None, domain_logits=None, domain_ids=None,
                prototypes=None, num_classes=3) -> dict:
        losses, total = {}, 0.0

        losses["ce"] = ce_loss(logits, labels)
        total = total + self.weights.get("ce", 1.0) * losses["ce"]

        if self.weights.get("supcon", 0.0) > 0:
            losses["supcon"] = supcon_loss(features, labels, target_ids, self.supcon_t)
            total = total + self.weights["supcon"] * losses["supcon"]

        if self.weights.get("proto", 0.0) > 0:
            if prototypes is None:
                prototypes = compute_prototypes(features, labels, num_classes)
            losses["proto"] = proto_loss(features, labels, prototypes, self.proto_t)
            total = total + self.weights["proto"] * losses["proto"]

        if self.weights.get("domain", 0.0) > 0 and domain_logits is not None:
            losses["domain"] = domain_loss(domain_logits, domain_ids)
            total = total + self.weights["domain"] * losses["domain"]

        losses["total"] = total
        return losses
