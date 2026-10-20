# -*- coding: utf-8 -*-
"""严格零样本划分。

- Subtask A：剔除训练集中与 val/test 重叠的目标（上游有少量模板化 claim 重叠）。
- Subtask B：测试域 = 留出域；训练/验证仅取其它 6 域且 in_use==1（过滤 In Use==0 的 LexsimTopics）。

产出 data/processed/splits/{A|B}/...parquet，供 dataset.py 与 check_protocol.py 使用。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import Config
from src.data.constants import DOMAIN_ORDER


def _write(df: pd.DataFrame, out: Path, split: str):
    sub = df[df["split"] == split].reset_index(drop=True)
    sub.to_parquet(out / f"{split}.parquet", index=False)


def build_subtaskA(all_df: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    a = all_df[all_df["subtask"] == "A"].copy()

    tr, va, te = a[a["split"] == "train"], a[a["split"] == "val"], a[a["split"] == "test"]
    te_t = set(te["target"])
    va_strict = va[~va["target"].isin(te_t)]          # val 不与 test 重叠
    va_t = set(va_strict["target"])
    tr_strict = tr[~tr["target"].isin(va_t | te_t)]   # train 不与 val/test 重叠
    print(f"[A] 严格化：val 剔除 {len(va) - len(va_strict)} 行（与 test 重叠），"
          f"train 剔除 {len(tr) - len(tr_strict)} 行（与 val/test 重叠）")

    _write(tr_strict, out, "train")
    _write(va_strict, out, "val")
    _write(te, out, "test")
    print(f"[A] 写出 {out}（train={len(tr_strict):,} val={len(va_strict):,} test={len(te):,}）")


def build_subtaskB(all_df: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    b = all_df[all_df["subtask"] == "B"].copy()

    for dom in DOMAIN_ORDER:
        fold = out / dom
        fold.mkdir(parents=True, exist_ok=True)

        # 测试 = 留出域（单域）；训练/验证 = 其它 6 域且 in_use==1
        test = b[(b["domain_fold"] == dom) & (b["split"] == "test")]
        src = b[(b["domain_fold"] == dom) & (b["split"].isin(["train", "val"]))]
        src = src[src["in_use"] == 1]  # 过滤 In Use==0（LexsimTopics）
        assert set(test["domain"].unique()) == {dom}, "测试集应为纯单域"
        assert dom not in set(src["domain"].unique()), "训练/验证不得包含留出域"

        _write(src, fold, "train")
        _write(src, fold, "val")
        _write(test, fold, "test")
        print(f"[B] {dom}: train={len(src[src['split']=='train']):,} "
              f"val={len(src[src['split']=='val']):,} test={len(test):,}")


def run(cfg: Config):
    all_df = pd.read_parquet(cfg.path("interim") / "all.parquet")
    splits = cfg.path("splits")
    build_subtaskA(all_df, splits / "A")
    build_subtaskB(all_df, splits / "B")
    print("划分完成。")


def load_split(cfg: Config, subtask: str, split: str, fold: str | None = None) -> pd.DataFrame:
    """加载某个划分：A 或 B（B 需指定留出域 fold）。"""
    base = cfg.path("splits") / subtask
    if subtask == "A":
        return pd.read_parquet(base / f"{split}.parquet")
    return pd.read_parquet(base / fold / f"{split}.parquet")


if __name__ == "__main__":
    run(Config.load())
