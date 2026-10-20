# -*- coding: utf-8 -*-
"""增强基类 + 三级质量闸门（标签一致性 / 去重 / 泄漏扫描）。

增强样本只能进训练集，且不得破坏零样本假设（不得含测试目标/测试域）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Augmenter(ABC):
    """所有增强方法的统一接口。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入训练集子表（含 text/target/stance_id 等列），返回新增的增强行（aug_style 已标）。"""


def quality_gates(df: pd.DataFrame, original: pd.DataFrame,
                  test_targets: set, test_domains: set,
                  label_checker=None) -> pd.DataFrame:
    """三级质量闸门，逐级过滤增强样本。

    - label_consistency：用 label_checker(text, target) 预测立场，与增强样本标签一致才保留
      （增强不改变 target 时，可直接继承原始标签，此步校验改写未翻转极性）。
    - dedup：与原始样本及增强内部去重（按 (text, target)）。
    - leak_scan：剔除包含测试目标或测试域的样本。
    """
    kept = df.copy()

    # ① 标签一致性
    if label_checker is not None:
        before = len(kept)
        mask = kept.apply(
            lambda r: label_checker(r["text"], r["target"]) == r["stance_id"], axis=1)
        kept = kept[mask]
        print(f"  [gate] label_consistency 剔除 {before - len(kept)} 行")

    # ② 去重
    before = len(kept)
    key = lambda d: d[["text", "target"]].agg(tuple, axis=1)
    seen = set(key(original))
    mask = []
    for _, r in kept.iterrows():
        k = (r["text"], r["target"])
        if k in seen:
            mask.append(False)
        else:
            seen.add(k)
            mask.append(True)
    kept = kept[mask]
    print(f"  [gate] dedup 剔除 {before - len(kept)} 行")

    # ③ 泄漏扫描
    before = len(kept)
    mask = (~kept["target"].isin(test_targets)) & (~kept["domain"].isin(test_domains))
    kept = kept[mask]
    print(f"  [gate] leak_scan 剔除 {before - len(kept)} 行")

    return kept.reset_index(drop=True)
