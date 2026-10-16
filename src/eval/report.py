# -*- coding: utf-8 -*-
"""结果报告：把各方法/各域/各折的指标汇总成表格（markdown / csv / latex）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_tables(results: dict, out_dir: Path):
    """results: {table_name: DataFrame}，写出 csv + markdown + latex。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
        try:
            (out_dir / f"{name}.tex").write_text(df.to_latex(index=False), encoding="utf-8")
        except Exception:
            pass
    print(f"[report] 表格已写出到 {out_dir}")
