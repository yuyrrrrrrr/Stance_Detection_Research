# -*- coding: utf-8 -*-
"""域对抗（DANN）：梯度反转层 + 域判别器（Ganin et al., JMLR 2016）。

本项目扩展：把「真实域」与「增强风格」拼成联合域标签，一起送入判别器，
使模型对真实领域和增强风格都保持不变性（见 configs/dg.yaml 的 domain.aug_styles）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None  # 反转梯度，第二项对应 lambda_


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """输入池化特征，输出域/风格 logits。"""

    def __init__(self, hidden_size: int, num_domains: int, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x):
        return self.net(x)


def ganin_lambda(progress: float, alpha: float = 10.0) -> float:
    """Ganin 调度：λ(p) = 2/(1+exp(-α·p)) - 1，p∈[0,1] 训练进度。"""
    import math
    return 2.0 / (1.0 + math.exp(-alpha * progress)) - 1.0
