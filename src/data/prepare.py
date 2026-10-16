# -*- coding: utf-8 -*-
"""数据预处理：把上游 24 个 CSV 归一化为统一 schema 的 parquet，并生成带校验值的清单。

统一长表 schema（data/interim/all.parquet）：
    split / subtask / left_out_domain / domain / text / target /
    label / label_id / target_type / target_id / domain_id / in_use / aug_style / source_file

运行：python -m src.data.prepare
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.common.config import Config
from src.data.constants import (DOMAIN_FOLDERS, DOMAIN_TO_ID, LABEL_TO_ID,
                                SPLIT_FILES, TYPE_MAP)

# 列名归一化映射（自动适配 Text/Target 1/Stance 1/Type/Domain/In Use 等变体）
_COL_RENAME = {
    "Text": "text", "text": "text", "内容": "text", "微博": "text",
    "Target 1": "target", "target": "target", "目标": "target",
    "Stance 1": "label", "label": "label", "立场": "label",
    "Type": "target_type", "type": "target_type", "类型": "target_type",
    "Domain": "domain", "domain": "domain", "领域": "domain",
    "In Use": "in_use", "in_use": "in_use", "是否使用": "in_use",
}


def _read_csv(path: Path, **meta) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=_COL_RENAME)
    for k, v in meta.items():
        df[k] = v
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip()
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    assert df["label_id"].notna().all(), f"存在未知标签：{df['label'].unique()}"
    df["target_type"] = (df["target_type"].astype(str).str.strip().str.lower()
                         .map(TYPE_MAP).fillna("unknown"))
    if "in_use" in df.columns:
        df["in_use"] = df["in_use"].astype(str).str.strip()
    df["aug_style"] = "orig"
    return df


def load_raw(cfg: Config) -> pd.DataFrame:
    """读取 subtaskA + subtaskB 全部 CSV，归一化后拼接。"""
    raw = cfg.raw_root
    frames = []

    # Subtask A（目标级零样本，无 Domain/In Use 列）
    a_root = raw / "subtaskA"
    for split, fname in SPLIT_FILES.items():
        df = _read_csv(a_root / fname, split=split, subtask="A", left_out_domain="")
        frames.append(df)

    # Subtask B（领域级零样本，每个文件夹 = 一个留出测试域）
    b_root = raw / "subtaskB"
    for folder, dom in DOMAIN_FOLDERS.items():
        for split, fname in SPLIT_FILES.items():
            df = _read_csv(b_root / folder / fname, split=split, subtask="B",
                           left_out_domain=dom)
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    return _normalize(all_df)


def _assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    """全局目标 id（跨划分一致，供 SupCon 区分同/不同目标）与领域 id。"""
    df = df.copy()
    df["target_id"] = pd.factorize(df["target"])[0]
    if "domain" in df.columns:
        df["domain_id"] = df["domain"].map(DOMAIN_TO_ID)
    else:
        df["domain_id"] = -1  # subtask A 无领域
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cfg: Config) -> Path:
    """主入口：产出 data/interim/all.parquet 与 manifest.json。"""
    interim = cfg.path("interim")
    interim.mkdir(parents=True, exist_ok=True)

    print("读取并归一化 24 个 CSV ...")
    df = load_raw(cfg)
    df = _assign_ids(df)

    out = interim / "all.parquet"
    df.to_parquet(out, index=False)
    print(f"已写出 {out}（{len(df):,} 行）")

    # 清单：源文件校验值 + 行数，保证可复现
    manifest = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sources": [
            {"file": str(f), "sha256": _sha256(f), "rows": int(len(_read_csv(f)))}
            for f in sorted(cfg.raw_root.rglob("*.csv"))
        ],
    }
    mpath = interim / "manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"已写出 {mpath}")
    return out


if __name__ == "__main__":
    run(Config.load())
