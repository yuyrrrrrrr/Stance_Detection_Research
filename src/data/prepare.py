# -*- coding: utf-8 -*-
"""数据清洗 + 预处理：把上游 24 个 CSV 归一化为单一 schema 的 parquet，并做质量闸门与去重。

统一长表 schema（data/interim/all.parquet）：
    row_id / subtask / domain_fold / split / domain / text / target /
    stance / stance_id / target_type / in_use / aug_style / source_file /
    target_id / domain_id
    其中 row_id 全局唯一（{subtask}_{domain_fold}_{split}_{行号}）。

清洗步骤（对应需求）：
  1. 列名统一：Text→text、Target 1→target、Stance 1→stance、Domain→domain、
     In Use→in_use、Type→target_type；补 row_id/subtask/domain_fold；
     Subtask A 补 domain="NA"、in_use=1。
  2. 标签编码：stance→stance_id，映射唯一来源是 configs/base.yaml（经 src/common/config.py）。
     出现未知标签直接报错，不静默丢弃。
  3. 去重：按 (subtask, domain_fold, text, target, stance) 完全去重（作用域说明见下）；
     额外报告「同 (text,target) 不同 stance」的冲突行数（标签噪声重点嫌疑）。
  4. 清洗：不改动文本（EDA 已确认无 @/URL/话题/emoji），仅空白归一化
     （全角空格/不换行空格→半角、零宽空格移除、连续空白折叠为单个空格）。
  5. 类型归一化：clauses→claim、noun_phrases→noun_phrase（内部二值字段 target_type）。
  6. 质量断言：无空 text/target；stance_id ∈ {0,1,2}；len(text) ≤ 200。

去重作用域（重要）：按 (subtask, domain_fold) 分组，**不是全局**。Subtask B 是 7 个
留一域 fold，同一份样本会按设计出现在多个 fold（48,132 个唯一样本 × 最多 7 fold
= 336,924 行）；全局去重会把留一域结构压成约 48k 行，破坏零样本划分。

运行：python -m src.data.prepare
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.common.config import Config
from src.data.constants import DOMAIN_FOLDERS, DOMAIN_TO_ID, SPLIT_FILES, TYPE_MAP

# 原始列名 -> 规范列名（已核实的实际列名；Domain/In Use 仅 Subtask B 存在）
_COL_RENAME = {
    "Text": "text",
    "Target 1": "target",
    "Stance 1": "stance",
    "Domain": "domain",
    "In Use": "in_use",
    "Type": "target_type",
}

# 空白归一化要处理的特殊字符（用 chr 显式定义，避免源码里出现不可见字符）
_ZERO_WIDTH_SPACE = chr(0x200B)   # 零宽空格（不可见）
_NO_BREAK_SPACE = chr(0x00A0)     # 不换行空格
_FULLWIDTH_SPACE = chr(0x3000)    # 全角空格


def _norm_ws(s: str) -> str:
    """空白归一化：零宽空格移除、不换行/全角空格→半角、连续空白折叠为单个空格并去首尾。"""
    if not isinstance(s, str):
        return ""
    s = s.replace(_ZERO_WIDTH_SPACE, "")
    s = s.replace(_NO_BREAK_SPACE, " ")
    s = s.replace(_FULLWIDTH_SPACE, " ")
    return " ".join(s.split())    # 折叠所有连续空白（含换行/制表）为单个半角空格


def _read_csv(path: Path, split: str, subtask: str, domain_fold: str,
              raw_root: Path) -> pd.DataFrame:
    """读取单个 CSV：去 BOM、统一列名、补 row_id/subtask/domain_fold/split/source_file。"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=_COL_RENAME)
    df["row_id"] = [f"{subtask}_{domain_fold}_{split}_{i:06d}" for i in range(len(df))]  # 全局唯一
    df["subtask"] = subtask
    df["domain_fold"] = domain_fold
    df["split"] = split
    df["source_file"] = path.relative_to(raw_root).as_posix()
    return df


def _encode_stance(df: pd.DataFrame, label_to_id: dict) -> pd.DataFrame:
    """标签编码：stance → stance_id。未知标签直接报错。"""
    ids = df["stance"].map(label_to_id)
    if ids.isna().any():
        unknown = sorted(set(df.loc[ids.isna(), "stance"].unique()))
        raise ValueError(f"存在未知标签（未在 configs/base.yaml 定义）：{unknown}")
    df["stance_id"] = ids.astype(int)
    return df


def _normalize_target_type(df: pd.DataFrame) -> pd.DataFrame:
    """类型归一化：clauses→claim、noun_phrases→noun_phrase（内部二值）。"""
    tt = df["target_type"].astype(str).str.strip().str.lower().map(TYPE_MAP)
    if tt.isna().any():
        unknown = sorted(set(df.loc[tt.isna(), "target_type"].astype(str).unique()))
        raise ValueError(f"存在未知 target_type：{unknown}")
    df["target_type"] = tt
    return df


def _assert_quality(df: pd.DataFrame) -> None:
    """质量断言：无空 text/target；stance_id ∈ {0,1,2}；len(text) ≤ 200。"""
    problems = []
    n_empty_text = int((df["text"] == "").sum())
    n_empty_target = int((df["target"] == "").sum())
    if n_empty_text:
        problems.append(f"空 text：{n_empty_text} 行")
    if n_empty_target:
        problems.append(f"空 target：{n_empty_target} 行")
    n_bad_stance = int((~df["stance_id"].isin([0, 1, 2])).sum())
    if n_bad_stance:
        problems.append(f"非法 stance_id：{n_bad_stance} 行")
    n_over = int((df["text"].str.len() > 200).sum())
    if n_over:
        problems.append(f"len(text)>200：{n_over} 行")
    if problems:
        raise ValueError("质量断言失败：" + "；".join(problems))


def _conflict_details(df: pd.DataFrame, conflict_index) -> list:
    """冲突明细：同 (text,target) 但 stance 不一致的样本，供人工核查标签噪声。"""
    details = []
    for txt, tgt in conflict_index:
        sub = df[(df["text"] == txt) & (df["target"] == tgt)]
        stance_rows = {k: int(v) for k, v in sub["stance"].value_counts().items()}
        details.append({
            "target": tgt,
            "text": txt,
            "stances": sorted(sub["stance"].unique().tolist()),
            "stance_rows": stance_rows,
            "source_files": sorted(sub["source_file"].unique().tolist()),
        })
    details.sort(key=lambda d: (d["target"], d["text"]))
    return details


def _clean(df: pd.DataFrame, cfg: Config):
    """对拼接后的长表执行清洗，返回 (df, report)。"""
    df = df.copy()

    # ① 空白归一化（text/target/stance，不改动文本内容）
    for col in ("text", "target", "stance"):
        df[col] = df[col].map(_norm_ws)

    # ② 标签编码（唯一来源 configs/base.yaml）
    df = _encode_stance(df, cfg.label_to_id)

    # ③ 类型归一化
    df = _normalize_target_type(df)

    # ④ 去重（作用域：subtask × domain_fold，保留留一域结构）
    n_before = len(df)
    df = df.drop_duplicates(
        subset=["subtask", "domain_fold", "text", "target", "stance"],
        keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)

    # ⑤ 冲突报告：同 (text,target) 不同 stance（全局；标签噪声嫌疑，保留待查）
    g = df.groupby(["text", "target"])["stance_id"].agg(["nunique", "count"])
    conflict = g[g["nunique"] > 1]
    conflict_pairs = int(len(conflict))
    conflict_rows = int(conflict["count"].sum())
    conflict_details = _conflict_details(df, conflict.index) if conflict_pairs else []

    # ⑥ 全局目标 id（跨划分一致，供 SupCon 区分同/不同目标）与领域 id
    df["target_id"] = pd.factorize(df["target"])[0]
    df["domain_id"] = df["domain"].map(DOMAIN_TO_ID).fillna(-1).astype(int)

    # ⑦ 增强风格占位（清洗产物均为原始样本）
    df["aug_style"] = "orig"

    # ⑧ 质量断言
    _assert_quality(df)

    cols = ["row_id", "subtask", "domain_fold", "split", "domain", "text", "target",
            "stance", "stance_id", "target_type", "in_use", "aug_style", "source_file",
            "target_id", "domain_id"]
    df = df[cols]

    report = {
        "raw_rows": n_before,
        "rows": int(len(df)),
        "dedup_removed": n_removed,
        "conflict_pairs": conflict_pairs,
        "conflict_rows": conflict_rows,
        "conflict_details": conflict_details,
    }
    return df, report


def _read_and_clean(cfg: Config):
    """读取 24 个 CSV 并清洗，返回 (df, report)。"""
    raw_root = cfg.raw_root
    frames = []

    # Subtask A：目标级零样本，无 Domain/In Use 列 -> 补 domain="NA"、in_use=1
    a_root = raw_root / "subtaskA"
    for split, fname in SPLIT_FILES.items():
        df = _read_csv(a_root / fname, split, "A", "NA", raw_root)
        df["domain"] = "NA"
        df["in_use"] = 1
        frames.append(df)

    # Subtask B：领域级零样本，每个文件夹 = 一个留出测试域
    b_root = raw_root / "subtaskB"
    for folder, dom in DOMAIN_FOLDERS.items():
        for split, fname in SPLIT_FILES.items():
            df = _read_csv(b_root / folder / fname, split, "B", dom, raw_root)
            df["in_use"] = df["in_use"].map({"0": 0, "1": 1})
            frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    return _clean(df, cfg)


def load_cleaned(cfg: Config) -> pd.DataFrame:
    """读取上游 24 个 CSV 并完成清洗，返回统一 schema 的 DataFrame（等价于 data/interim/all.parquet）。"""
    return _read_and_clean(cfg)[0]


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

    print("读取并清洗 24 个 CSV ...")
    df, report = _read_and_clean(cfg)

    out = interim / "all.parquet"
    df.to_parquet(out, index=False)

    sources = []
    for p in sorted(cfg.raw_root.rglob("*.csv")):
        sources.append({
            "file": p.relative_to(cfg.raw_root).as_posix(),
            "sha256": _sha256(p),
            "rows": int(pd.read_csv(p, encoding="utf-8-sig", dtype=str).shape[0]),
        })

    manifest = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "cleaning": {k: v for k, v in report.items() if k != "conflict_details"},
        "sources": sources,
    }
    mpath = interim / "manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 冲突待查清单：同 (text,target) 不同 stance 的标签噪声嫌疑样本（独立产物，供人工核查）
    conflicts = {
        "description": "同 (text,target) 但立场(stance)标注不一致的样本（标签噪声重点嫌疑，保留未删除）",
        "n_pairs": report["conflict_pairs"],
        "n_rows": report["conflict_rows"],
        "pairs": report["conflict_details"],
    }
    cpath = interim / "conflicts.json"
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(conflicts, f, ensure_ascii=False, indent=2)

    print(f"[清洗] 原始行数={report['raw_rows']:,}  输出行数={report['rows']:,}  "
          f"去重移除={report['dedup_removed']:,}")
    print(f"[冲突] 同 (text,target) 不同 stance：{report['conflict_pairs']} 对 / "
          f"{report['conflict_rows']} 行（保留待查，已写出清单）")
    print("[质量] 全部通过：无空 text/target、stance_id∈{0,1,2}、len(text)≤200")
    print(f"已写出 {out}")
    print(f"已写出 {mpath}")
    print(f"已写出 {cpath}")
    return out


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(Config.load())
