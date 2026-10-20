# -*- coding: utf-8 -*-
"""阶段 3 划分的最小协议测试。

运行：pytest -q tests/test_split.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import Config
from src.data.constants import DOMAIN_ORDER

ROOT = Path(__file__).resolve().parents[1]
cfg = Config.load()
SPLITS = cfg.path("splits")


def _load(p):
    return pd.read_parquet(p)


def test_A_strict_target_disjoint():
    tr = _load(SPLITS / "A" / "train.parquet")
    va = _load(SPLITS / "A" / "val.parquet")
    te = _load(SPLITS / "A" / "test.parquet")
    assert set(tr["target"]) & set(va["target"]) == set()
    assert set(tr["target"]) & set(te["target"]) == set()
    assert set(va["target"]) & set(te["target"]) == set()


def test_A_dev_no_leak():
    tr = _load(SPLITS / "A" / "train.parquet")
    dev = _load(SPLITS / "A" / "dev.parquet")
    va = _load(SPLITS / "A" / "val.parquet")
    te = _load(SPLITS / "A" / "test.parquet")
    # dev 从 train 切出，与 train/val/test 的 row 均不交叉
    assert set(dev["row_id"]) & set(tr["row_id"]) == set()
    assert set(dev["row_id"]) & set(va["row_id"]) == set()
    assert set(dev["row_id"]) & set(te["row_id"]) == set()
    # strict 下 dev 目标不与 val/test 重叠
    assert set(dev["target"]) & set(va["target"]) == set()
    assert set(dev["target"]) & set(te["target"]) == set()


def test_B_test_domain_not_in_train_val():
    for dom in DOMAIN_ORDER:
        tr = _load(SPLITS / "B" / dom / "train.parquet")
        va = _load(SPLITS / "B" / dom / "val.parquet")
        te = _load(SPLITS / "B" / dom / "test.parquet")
        assert set(te["domain"]) == {dom}
        assert dom not in set(tr["domain"])
        assert dom not in set(va["domain"])
        assert set(tr["domain"]) == set(va["domain"])


def test_B_inuse_filter():
    for dom in DOMAIN_ORDER:
        tr = _load(SPLITS / "B" / dom / "train.parquet")
        va = _load(SPLITS / "B" / dom / "val.parquet")
        te = _load(SPLITS / "B" / dom / "test.parquet")
        assert (te["in_use"] == 1).all()
        assert (tr["in_use"] == 1).all()
        assert (va["in_use"] == 1).all()


def test_row_id_global_unique():
    df = pd.read_parquet(cfg.path("interim") / "all.parquet")
    assert df["row_id"].is_unique, "row_id 必须全局唯一"


def test_split_hashes_match_manifest():
    mpath = SPLITS.parent / "split_manifest.json"
    assert mpath.exists(), "缺少 split_manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    for entry in m["files"]:
        fpath = ROOT / entry["file"]
        assert fpath.exists(), f"缺失 {entry['file']}"
        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
        assert actual == entry["sha256"], f"哈希不一致 {entry['file']}"
        assert len(_load(fpath)) == entry["rows"], f"行数不一致 {entry['file']}"
