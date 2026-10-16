# -*- coding: utf-8 -*-
"""统计检验：跨随机种子聚合（均值±标准差）+ Wilcoxon 符号秩检验 + Holm 校正。

协议（见 README 九）：至少 5 个随机种子；与基线配对做 Wilcoxon 符号秩检验，Holm 校正多重比较。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def summarize(scores: Sequence[float]) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    return {"mean": float(scores.mean()), "std": float(scores.std(ddof=1)),
            "min": float(scores.min()), "max": float(scores.max()), "n": int(len(scores))}


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]):
    """配对 Wilcoxon 符号秩检验（双尾），返回 (statistic, p_value)。"""
    stat, p = stats.wilcoxon(np.asarray(a, float), np.asarray(b, float))
    return float(stat), float(p)


def holm_correction(pvalues: Sequence[float]) -> List[float]:
    """Holm 逐步下降校正。"""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p[idx] * (n - rank))
    # 单调化
    for i in range(1, n):
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i - 1]])
    return adjusted.tolist()


def compare_methods(baseline_scores: Sequence[float],
                    methods: Dict[str, Sequence[float]]) -> pd.DataFrame:
    """返回每个方法 vs 基线的均值±标准差 + Wilcoxon p 值 + Holm 校正 p 值。"""
    import pandas as pd
    rows = []
    pvals = []
    for name, scores in methods.items():
        s = summarize(scores)
        _, p = wilcoxon_signed_rank(scores, baseline_scores)
        rows.append({"method": name, "mean": s["mean"], "std": s["std"], "p": p})
        pvals.append(p)
    adj = holm_correction(pvals)
    for r, ap in zip(rows, adj):
        r["p_holm"] = ap
    return pd.DataFrame(rows)
