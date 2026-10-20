# -*- coding: utf-8 -*-
"""阶段 3：严格零样本划分（可复现、可断言、无测试泄漏）。

Subtask A（目标级零样本）：
  - paper  ：保留官方原始 train/val/test，从 train 切 5% dev（按 stance_id 分层）。
  - strict ：test 不变；从 train/val 删除 test 目标，再从 train 删除 val 目标，
             使 train/val/test 目标两两不交；从 strict train 切 5% dev（按 stance_id 分层）。

Subtask B（领域级零样本）：7 个留一域 fold；
  test = 留出域（in_use 全 1）；train/val = 其余 6 域，仅过滤 in_use==0。

产出：
  data/processed/splits/A/{train,dev,val,test}.parquet          # strict，主实验
  data/processed/splits/A/paper/{train,dev,val,test}.parquet    # paper，附录对照
  data/processed/splits/B/{domain}/{train,val,test}.parquet     # 7 个留一域 fold
  data/processed/split_manifest.json

说明：
  - 清洗已在 prepare.py 完成，本模块只做划分，不生成任何增强样本（aug_style 全为 orig）。
  - Subtask B 的 7 个 fold 是独立实验，按设计重复包含底层样本，不做全局去重。
  - row_id 在 prepare.py 中已全局唯一。

运行：python -m src.data.split
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import Config
from src.data.constants import DOMAIN_ORDER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_FRAC = 0.05
SEED = 42


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _dist_int(series: pd.Series) -> dict:
    return {int(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _dist_str(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _write(df: pd.DataFrame, path: Path) -> dict:
    """写 parquet 并返回文件信息（相对路径 / sha256 / 行数 / target 数 / 分布）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    df.to_parquet(path, index=False)
    return {
        "file": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256(path),
        "rows": int(len(df)),
        "n_target": int(df["target"].nunique()),
        "stance_id_dist": _dist_int(df["stance_id"]),
        "domain_dist": _dist_str(df["domain"]),
    }


def _cut_dev(train: pd.DataFrame, seed: int, frac: float):
    """从 train 切出 frac 比例的 dev（按 stance_id 分层），返回 (train_core, dev)。"""
    train_core, dev = train_test_split(
        train, test_size=frac, random_state=seed, stratify=train["stance_id"])
    return train_core.reset_index(drop=True), dev.reset_index(drop=True)


def _overlap(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame) -> dict:
    """目标两两交集统计。"""
    return {
        "train_val": int(len(set(tr["target"]) & set(va["target"]))),
        "train_test": int(len(set(tr["target"]) & set(te["target"]))),
        "val_test": int(len(set(va["target"]) & set(te["target"]))),
    }


def build_subtaskA(a: pd.DataFrame, out: Path) -> dict:
    """构建 paper 与 strict 两个版本，返回 manifest 片段。"""
    te = a[a["split"] == "test"].copy()
    tr_orig = a[a["split"] == "train"].copy()
    va_orig = a[a["split"] == "val"].copy()

    # ---- paper：保留官方原始划分，仅切 dev ----
    tr_paper, dev_paper = _cut_dev(tr_orig, SEED, DEV_FRAC)

    # ---- strict：test 不变；train/val 与 test 目标不交；train 与 val 不交 ----
    te_t = set(te["target"])
    va_strict = va_orig[~va_orig["target"].isin(te_t)]
    tr_strict_full = tr_orig[~tr_orig["target"].isin(te_t)]
    va_t = set(va_strict["target"])
    tr_strict_full = tr_strict_full[~tr_strict_full["target"].isin(va_t)]
    tr_strict, dev_strict = _cut_dev(tr_strict_full, SEED, DEV_FRAC)

    # strict 目标两两不交（精确串相等，单轮即收敛；此处显式断言兜底）
    for name, (x, y) in {"train_val": (tr_strict, va_strict),
                         "train_test": (tr_strict, te),
                         "val_test": (va_strict, te)}.items():
        assert len(set(x["target"]) & set(y["target"])) == 0, f"A strict {name} 目标交集非空"

    manifest = {}
    s = out / "A"
    manifest["strict"] = {
        "train": _write(tr_strict, s / "train.parquet"),
        "dev": _write(dev_strict, s / "dev.parquet"),
        "val": _write(va_strict, s / "val.parquet"),
        "test": _write(te, s / "test.parquet"),
        "target_overlap": _overlap(tr_strict, va_strict, te),
        "dropped": {
            "val_from_test": int(len(va_orig) - len(va_strict)),
            "train_from_val_test": int(len(tr_orig) - len(tr_strict_full)),
        },
    }
    p = out / "A" / "paper"
    manifest["paper"] = {
        "train": _write(tr_paper, p / "train.parquet"),
        "dev": _write(dev_paper, p / "dev.parquet"),
        "val": _write(va_orig, p / "val.parquet"),
        "test": _write(te, p / "test.parquet"),
        # 报告官方原始重叠（切 dev 前的 train），用于与论文数字对照
        "target_overlap": _overlap(tr_orig, va_orig, te),
    }
    return manifest


def build_subtaskB(b: pd.DataFrame, out: Path) -> dict:
    """构建 7 个留一域 fold，返回 manifest 片段。"""
    manifest = {}
    for dom in DOMAIN_ORDER:
        fold = out / dom
        test = b[(b["domain_fold"] == dom) & (b["split"] == "test")]
        tr_raw = b[(b["domain_fold"] == dom) & (b["split"] == "train")]
        va_raw = b[(b["domain_fold"] == dom) & (b["split"] == "val")]
        tr = tr_raw[tr_raw["in_use"] == 1]
        va = va_raw[va_raw["in_use"] == 1]

        # 协议断言
        assert set(test["domain"].unique()) == {dom}, f"B {dom}: test 非纯单域"
        assert (test["in_use"] == 1).all(), f"B {dom}: test 含 in_use!=1"
        assert dom not in set(tr["domain"].unique()), f"B {dom}: 训练集含留出域"
        assert dom not in set(va["domain"].unique()), f"B {dom}: 验证集含留出域"
        assert set(tr["domain"].unique()) == set(va["domain"].unique()), \
            f"B {dom}: train/val 域集合不一致"

        manifest[dom] = {
            "train": _write(tr, fold / "train.parquet"),
            "val": _write(va, fold / "val.parquet"),
            "test": _write(test, fold / "test.parquet"),
            "test_domain": sorted(test["domain"].unique().tolist()),
            "train_domains": sorted(tr["domain"].unique().tolist()),
            "val_domains": sorted(va["domain"].unique().tolist()),
            "dropped_in_use_0": {
                "train": int(len(tr_raw) - len(tr)),
                "val": int(len(va_raw) - len(va)),
            },
        }
    return manifest


def run(cfg: Config) -> Path:
    all_parquet = cfg.path("interim") / "all.parquet"
    all_df = pd.read_parquet(all_parquet)
    splits = cfg.path("splits")

    # 清理旧划分，保证无残留（保留 .gitkeep）
    if splits.exists():
        shutil.rmtree(splits)
    splits.mkdir(parents=True)
    (splits / ".gitkeep").touch()

    a = all_df[all_df["subtask"] == "A"]
    b = all_df[all_df["subtask"] == "B"]

    a_manifest = build_subtaskA(a, splits)
    b_manifest = build_subtaskB(b, splits / "B")

    # 平铺文件清单（用于哈希/行数一致性校验）
    files = []
    for section in (a_manifest["strict"], a_manifest["paper"]):
        for k in ("train", "dev", "val", "test"):
            info = section[k]
            files.append({"file": info["file"], "sha256": info["sha256"], "rows": info["rows"]})
    for dom in DOMAIN_ORDER:
        for k in ("train", "val", "test"):
            info = b_manifest[dom][k]
            files.append({"file": info["file"], "sha256": info["sha256"], "rows": info["rows"]})

    manifest = {
        "protocol_version": "3.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "python -m src.data.split",
        "git_commit": _git_commit(),
        "seed": SEED,
        "dev_frac": DEV_FRAC,
        "input": {
            "file": all_parquet.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(all_parquet),
            "rows": int(len(all_df)),
        },
        "口径说明": (
            "Subtask A：strict 为主实验（train/val/test 目标两两不交），paper 保留官方原始划分用于附录对照。"
            "Subtask B：7 个留一域 fold，test=留出域且 in_use 全 1，train/val=其余 6 域且过滤 in_use==0；"
            "7 折按设计重复包含底层样本，不做全局去重。dev 仅从 train 切 5%（按 stance_id 分层，seed=42）。"
            "本阶段不生成任何增强样本（aug_style 全为 orig）。"
        ),
        "A": a_manifest,
        "B": b_manifest,
        "files": files,
    }
    mpath = splits.parent / "split_manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"[A strict] train={a_manifest['strict']['train']['rows']:,} "
          f"dev={a_manifest['strict']['dev']['rows']:,} "
          f"val={a_manifest['strict']['val']['rows']:,} "
          f"test={a_manifest['strict']['test']['rows']:,} "
          f"重叠={a_manifest['strict']['target_overlap']}")
    print(f"[A paper ] train={a_manifest['paper']['train']['rows']:,} "
          f"dev={a_manifest['paper']['dev']['rows']:,} "
          f"val={a_manifest['paper']['val']['rows']:,} "
          f"test={a_manifest['paper']['test']['rows']:,} "
          f"重叠={a_manifest['paper']['target_overlap']}")
    for dom in DOMAIN_ORDER:
        info = b_manifest[dom]
        print(f"[B {dom}] train={info['train']['rows']:,} val={info['val']['rows']:,} "
              f"test={info['test']['rows']:,} dropped_in_use_0={info['dropped_in_use_0']}")
    print(f"已写出 {mpath}")
    return mpath


def load_split(cfg: Config, subtask: str, split: str, fold: str | None = None) -> pd.DataFrame:
    """加载某个划分：A 默认读 strict（A/{split}.parquet）；B 需指定留出域 fold。

    注意：paper 版本在 A/paper/ 下，不会被本函数默认读取（训练入口不会误用）。
    """
    base = cfg.path("splits") / subtask
    if subtask == "A":
        return pd.read_parquet(base / f"{split}.parquet")
    return pd.read_parquet(base / fold / f"{split}.parquet")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(Config.load())
