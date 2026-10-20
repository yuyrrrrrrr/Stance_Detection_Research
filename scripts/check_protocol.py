# -*- coding: utf-8 -*-
"""零样本协议一致性检查（阶段 3 冻结口径）。

检查内容：
  [1] 原始划分报告（上游数据固有属性，只读原始 CSV）
  [2] Subtask A strict 断言（目标两两不交、dev 不泄漏、无 row_id 交叉、无相同 text）
  [3] Subtask A paper 原始重叠报告（对照用，不做 PASS/FAIL）
  [4] Subtask B 7 折断言（测试域留出、in_use 过滤、域集合一致）
  [5] manifest 文件哈希一致性

运行：python scripts/check_protocol.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import Config
from src.data.constants import DOMAIN_FOLDERS, DOMAIN_ORDER, SPLIT_FILES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_raw_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str)


def _load(p):
    return pd.read_parquet(p)


def report_raw(cfg):
    """[1] 原始划分报告（上游数据固有属性）。"""
    raw = cfg.raw_root
    a = {}
    for split, fname in SPLIT_FILES.items():
        a[split] = set(_read_raw_csv(raw / "subtaskA" / fname)["Target 1"])
    print("=" * 70)
    print("[1] 原始划分报告（上游数据固有属性）")
    print("=" * 70)
    print(f"  Subtask A 目标重叠：train∩val={len(a['train'] & a['val'])}  "
          f"train∩test={len(a['train'] & a['test'])}  val∩test={len(a['val'] & a['test'])}")
    print("    -> 少量重叠多为模板化 claim，严格划分（split.py）会剔除。")
    for folder, dom in DOMAIN_FOLDERS.items():
        n = 0
        for split in ("train", "val"):
            df = _read_raw_csv(raw / "subtaskB" / folder / SPLIT_FILES[split])
            n += int((df["In Use"] == "0").sum())
        print(f"  Subtask B {dom}: 原始 train/val 含 In Use==0 = {n}")


def verify_A_strict(cfg):
    """[2] Subtask A strict 断言。"""
    base = cfg.path("splits") / "A"
    tr = _load(base / "train.parquet")
    dev = _load(base / "dev.parquet")
    va = _load(base / "val.parquet")
    te = _load(base / "test.parquet")

    print("=" * 70)
    print("[2] Subtask A strict 断言")
    print("=" * 70)
    checks = [
        ("train∩val target = 0", len(set(tr["target"]) & set(va["target"])) == 0),
        ("train∩test target = 0", len(set(tr["target"]) & set(te["target"])) == 0),
        ("val∩test target = 0", len(set(va["target"]) & set(te["target"])) == 0),
        ("dev 来自 train（dev∩train row 不交叉）", len(set(dev["row_id"]) & set(tr["row_id"])) == 0),
        ("dev 不泄漏 val", len(set(dev["row_id"]) & set(va["row_id"])) == 0
         and len(set(dev["target"]) & set(va["target"])) == 0),
        ("dev 不泄漏 test", len(set(dev["row_id"]) & set(te["row_id"])) == 0
         and len(set(dev["target"]) & set(te["target"])) == 0),
        ("train/test 无完全相同 text", len(set(tr["text"]) & set(te["text"])) == 0),
    ]
    # 四者 row_id 两两不交叉
    sets = {"train": set(tr["row_id"]), "dev": set(dev["row_id"]),
            "val": set(va["row_id"]), "test": set(te["row_id"])}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = names[i], names[j]
            checks.append((f"row_id 无交叉 {a_}/{b_}", len(sets[a_] & sets[b_]) == 0))

    ok = True
    for name, passed in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return ok


def verify_A_paper(cfg):
    """[3] Subtask A paper 原始重叠报告（对照，不做 PASS/FAIL）。"""
    base = cfg.path("splits") / "A" / "paper"
    tr = _load(base / "train.parquet")
    dev = _load(base / "dev.parquet")
    va = _load(base / "val.parquet")
    te = _load(base / "test.parquet")
    # 用 train ∪ dev 还原「切 dev 前」的完整官方 train，报告原始重叠
    tr_t = set(tr["target"]) | set(dev["target"])
    print("=" * 70)
    print("[3] Subtask A paper 原始重叠（对照用）")
    print("=" * 70)
    print(f"  train∪dev ∩ val={len(tr_t & set(va['target']))}  "
          f"train∪dev ∩ test={len(tr_t & set(te['target']))}  "
          f"val ∩ test={len(set(va['target']) & set(te['target']))}")
    print("    -> paper 版保留官方原始划分，存在重叠属上游固有属性。")


def verify_B(cfg):
    """[4] Subtask B 7 折断言。"""
    print("=" * 70)
    print("[4] Subtask B 7 折断言")
    print("=" * 70)
    ok = True
    for dom in DOMAIN_ORDER:
        base = cfg.path("splits") / "B" / dom
        tr = _load(base / "train.parquet")
        va = _load(base / "val.parquet")
        te = _load(base / "test.parquet")
        conds = {
            "测试域单域": set(te["domain"]) == {dom},
            "test in_use 全 1": (te["in_use"] == 1).all(),
            "留出域不在 train": dom not in set(tr["domain"]),
            "留出域不在 val": dom not in set(va["domain"]),
            "train/val 域集合一致": set(tr["domain"]) == set(va["domain"]),
            "train/val in_use 全 1": (tr["in_use"] == 1).all() and (va["in_use"] == 1).all(),
            "无 row_id 交叉": (len(set(tr["row_id"]) & set(va["row_id"])) == 0
                           and len(set(tr["row_id"]) & set(te["row_id"])) == 0
                           and len(set(va["row_id"]) & set(te["row_id"])) == 0),
        }
        fold_ok = all(conds.values())
        ok &= fold_ok
        bad = [k for k, v in conds.items() if not v]
        print(f"  Subtask B {dom}: 测试域={sorted(te['domain'].unique())} "
              f"训练域数={len(set(tr['domain']))} [{'PASS' if fold_ok else 'FAIL ' + str(bad)}]")
    return ok


def verify_manifest(cfg):
    """[5] manifest 文件哈希一致性。"""
    print("=" * 70)
    print("[5] manifest 文件哈希一致性")
    print("=" * 70)
    mpath = cfg.path("splits").parent / "split_manifest.json"
    if not mpath.exists():
        print("  [FAIL] split_manifest.json 缺失")
        return False
    m = json.loads(mpath.read_text(encoding="utf-8"))
    ok = True
    for entry in m.get("files", []):
        fpath = PROJECT_ROOT / entry["file"]
        match = fpath.exists() and (_sha256(fpath) == entry["sha256"]) \
            and (len(_load(fpath)) == entry["rows"])
        ok &= match
        if not match:
            print(f"  [FAIL] {entry['file']}")
    if ok:
        print(f"  [PASS] {len(m.get('files', []))} 个文件的 sha256/行数均与 manifest 一致")
    return ok


def main():
    cfg = Config.load()
    report_raw(cfg)
    print()
    ok = True
    ok &= verify_A_strict(cfg)
    print()
    verify_A_paper(cfg)
    print()
    ok &= verify_B(cfg)
    print()
    ok &= verify_manifest(cfg)
    print("=" * 70)
    print(f"[结论] {'全部 PASS（严格零样本协议成立）' if ok else '存在 FAIL，见上'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
