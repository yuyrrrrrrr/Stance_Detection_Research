# -*- coding: utf-8 -*-
"""零样本协议一致性检查（README 第九节的口径冻结检查）。

分两步：
1) 报告【原始划分】的已知问题（Subtask A 目标重叠、Subtask B 的 In Use==0 样本量）——供论文说明。
2) 验证【严格划分】（data/processed/splits/）是否满足零样本协议——真正的 PASS/FAIL。

运行：python scripts/check_protocol.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import Config
from src.data.constants import DOMAIN_ORDER
from src.data.prepare import load_raw


def report_raw(cfg: Config):
    """报告原始划分的已知问题（这些是上游数据的固有属性，不是本项目的错误）。"""
    df = load_raw(cfg)
    a = df[df["subtask"] == "A"]
    tr = set(a[a["split"] == "train"]["target"])
    va = set(a[a["split"] == "val"]["target"])
    te = set(a[a["split"] == "test"]["target"])
    print("=" * 70)
    print("[1] 原始划分报告（上游数据固有属性）")
    print("=" * 70)
    print(f"  Subtask A 目标重叠：train∩val={len(tr & va)}  train∩test={len(tr & te)}  "
          f"val∩test={len(va & te)}")
    print("    -> 少量重叠多为模板化 claim，严格划分（split.py）会剔除。")
    for dom in DOMAIN_ORDER:
        src = df[(df["subtask"] == "B") & (df["left_out_domain"] == dom)
                 & (df["split"].isin(["train", "val"]))]
        print(f"  Subtask B {dom}: 原始 train/val 含 In Use==0 = {int((src['in_use'] == '0').sum())}")


def verify_strict(cfg: Config):
    """验证严格划分满足零样本协议。"""
    splits = cfg.path("splits")
    ok = True

    # ---- Subtask A：目标互不相交 ----
    a = splits / "A"
    if a.exists():
        tr = set(pd.read_parquet(a / "train.parquet")["target"])
        va = set(pd.read_parquet(a / "val.parquet")["target"])
        te = set(pd.read_parquet(a / "test.parquet")["target"])
        a_ok = (len(tr & te) == 0) and (len(tr & va) == 0) and (len(va & te) == 0)
        ok &= a_ok
        print(f"  Subtask A 目标互不相交: train∩test={len(tr & te)} "
              f"[{'PASS' if a_ok else 'FAIL'}]")
    else:
        print("  Subtask A 划分缺失（先运行 python -m src.data.split）"); ok = False

    # ---- Subtask B：测试域留出 + In Use==0 过滤 ----
    for dom in DOMAIN_ORDER:
        fold = splits / "B" / dom
        if not fold.exists():
            print(f"  Subtask B {dom}: 划分缺失"); ok = False; continue
        tr = pd.read_parquet(fold / "train.parquet")
        te = pd.read_parquet(fold / "test.parquet")
        test_dom = set(te["domain"].unique())
        train_dom = set(tr["domain"].unique())
        n_inuse0 = int((tr["in_use"] == "0").sum())
        cond = (test_dom == {dom}) and (dom not in train_dom) and (n_inuse0 == 0)
        ok &= cond
        print(f"  Subtask B {dom}: 测试域={test_dom} 训练域数={len(train_dom)} "
              f"含留出域={'是' if dom in train_dom else '否'} InUse==0={n_inuse0} "
              f"[{'PASS' if cond else 'FAIL'}]")

    print("=" * 70)
    print(f"[结论] {'全部 PASS（严格零样本协议成立）' if ok else '存在 FAIL，见上'}")
    print("=" * 70)
    return ok


def main():
    cfg = Config.load()
    report_raw(cfg)
    print()
    print("=" * 70)
    print("[2] 严格划分验证（data/processed/splits/）")
    print("=" * 70)
    verify_strict(cfg)


if __name__ == "__main__":
    main()
