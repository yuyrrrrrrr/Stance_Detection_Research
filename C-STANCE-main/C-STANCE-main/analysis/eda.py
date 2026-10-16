# -*- coding: utf-8 -*-
"""
EDA for C-STANCE dataset, oriented toward the research direction:
    Domain Generalization (DG) + Data Augmentation (DA) for Chinese Zero-Shot Stance Detection.

Focus questions:
  1. Overall scale & split structure (subtask A vs B).
  2. Stance label distribution -> class imbalance (motivation for DA).
  3. Target analysis -> distinct targets, overlap across splits (zero-shot check),
     long-tail frequency (motivation for DA), target type (noun_phrases vs claims).
  4. Text analysis -> text length, #targets per text.
  5. Domain shift (subtask B) -> label distribution shift, cross-domain target overlap,
     vocabulary overlap, "In Use"==0 filtering volume (motivation for DG).
"""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "c_stance_dataset")
OUT = os.path.join(ROOT, "analysis", "output")
os.makedirs(OUT, exist_ok=True)

SUB_A = os.path.join(DATA, "subtaskA")
SUB_B = os.path.join(DATA, "subtaskB")

# Chinese stance labels
LABELS = ["支持", "反对", "中立"]
# Subtask B domain folders (folder name -> short Chinese domain used in 'Domain' col)
DOMAINS = {
    "新冠疫情_covid_epidemic": "新冠疫情",
    "世界事件_world_event": "世界事件",
    "文化教育_culture_and_education": "文化教育",
    "娱乐和消费_entertainment_and_consumption": "娱乐和消费",
    "体育_sports": "体育",
    "权益_rights": "权益",
    "环保_enviromental_protection": "环保",
}


def load(path, cols=None):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if cols:
        df = df[cols]
    return df


def bar(counts, title):
    print(f"\n  {title}")
    total = sum(counts.values())
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"    {k:>12}  {counts[k]:>7}  ({counts[k]/total*100:5.2f}%)")
    return total


def section(n, title):
    print("\n" + "=" * 78)
    print(f"[{n}] {title}")
    print("=" * 78)


def main():
    # ------------------------------------------------------------------ #
    section(1, "SUBTASK A — target-based zero-shot (scale & label distribution)")
    # ------------------------------------------------------------------ #
    for split in ["raw_train_all_onecol.csv", "raw_val_all_onecol.csv", "raw_test_all_onecol.csv"]:
        df = load(os.path.join(SUB_A, split))
        print(f"\n--- {split} : {len(df):,} rows ---")
        bar(Counter(df["Stance 1"]), "stance distribution")
        bar(Counter(df["Type"]), "target type (noun_phrases / claims)")

    train = load(os.path.join(SUB_A, "raw_train_all_onecol.csv"))
    val = load(os.path.join(SUB_A, "raw_val_all_onecol.csv"))
    test = load(os.path.join(SUB_A, "raw_test_all_onecol.csv"))

    # ------------------------------------------------------------------ #
    section(2, "SUBTASK A — target analysis (zero-shot & long-tail)")
    # ------------------------------------------------------------------ #
    tr_t = set(train["Target 1"]); va_t = set(val["Target 1"]); te_t = set(test["Target 1"])
    print(f"\ndistinct targets:  train={len(tr_t):,}  val={len(va_t):,}  test={len(te_t):,}")
    print(f"target overlap train∩test = {len(tr_t & te_t):,}  "
          f"train∩val = {len(tr_t & va_t):,}  val∩test = {len(va_t & te_t):,}")
    print("  -> zero-shot holds if train∩test == 0")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        freq = Counter(df["Target 1"])
        f = np.array(list(freq.values()))
        print(f"\n  [{name}] target frequency: total rows={len(df):,}, distinct={len(freq):,}")
        print(f"    mean={f.mean():.2f}  median={np.median(f):.0f}  max={f.max():.0f}")
        for thr in [1, 2, 3, 5, 10]:
            rare = (f <= thr).sum()
            print(f"    targets appearing <= {thr:>2}x : {rare:>6,} ({rare/len(freq)*100:5.2f}%)")
        # examples per target for a given label is always 1 (onecol), frequency==#texts per target

    # target length (chars) by type
    for name, df in [("train", train), ("test", test)]:
        df = df.copy()
        df["tlen"] = df["Target 1"].str.len()
        print(f"\n  [{name}] target char length: mean={df['tlen'].mean():.1f}  "
              f"median={df['tlen'].median():.0f}  max={df['tlen'].max():.0f}")
        print("    by type:", {t: f"{int(g['tlen'].mean())}" for t, g in df.groupby('Type')})

    # ------------------------------------------------------------------ #
    section(3, "SUBTASK A — text analysis")
    # ------------------------------------------------------------------ #
    for name, df in [("train", train), ("val", val), ("test", test)]:
        tlen = df["Text"].str.len()
        n_targets = df.groupby("Text")["Target 1"].nunique()
        print(f"\n  [{name}] text char length: mean={tlen.mean():.1f}  median={tlen.median():.0f}  max={tlen.max():.0f}")
        print(f"           distinct texts={df['Text'].nunique():,}  rows={len(df):,}  "
              f"avg targets/text={len(df)/df['Text'].nunique():.2f}")
        print(f"           #texts by #targets: {dict(Counter(n_targets.values))}")

    # label x type cross-tab (train) -> where augmentation may help
    print("\n  [train] stance x target-type cross-tab:")
    print(pd.crosstab(train["Stance 1"], train["Type"]))

    # ------------------------------------------------------------------ #
    section(4, "SUBTASK B — domain structure & label distribution shift")
    # ------------------------------------------------------------------ #
    print("\n(Each folder = one left-out test domain; the other 6 domains form train/val.)")
    for folder, dom in DOMAINS.items():
        tr = load(os.path.join(SUB_B, folder, "raw_train_all_onecol.csv"))
        va = load(os.path.join(SUB_B, folder, "raw_val_all_onecol.csv"))
        te = load(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        # In Use == 0 rows are filtered out during train/val
        tr_used = tr[tr["In Use"] == "1"]
        va_used = va[va["In Use"] == "1"]
        n_inuse0 = len(tr[tr["In Use"] == "0"]) + len(va[va["In Use"] == "0"])
        print(f"\n  LEFT-OUT domain = {dom} ({folder})")
        print(f"    train: {len(tr_used):,} used / {len(tr):,} total  |  "
              f"val: {len(va_used):,} used / {len(va):,} total  |  test: {len(te):,}")
        print(f"    rows with In Use==0 (Lexsim, dropped): {n_inuse0:,}")
        tr_dom = Counter(tr_used["Domain"]); te_dom = Counter(te["Domain"])
        print(f"    train domain mix: {dict(tr_dom)}")
        print(f"    test  domain mix: {dict(te_dom)}")

    # ------------------------------------------------------------------ #
    section(5, "SUBTASK B — label distribution shift across domains (DG)")
    # ------------------------------------------------------------------ #
    print("\nStance distribution per domain (using each domain's own TEST split, which is pure single-domain):")
    rows = []
    for folder, dom in DOMAINS.items():
        te = load(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        c = Counter(te["Stance 1"])
        tot = len(te)
        rows.append((dom, tot, c.get("支持", 0), c.get("反对", 0), c.get("中立", 0)))
    hdr = f"\n  {'domain':<10}{'n':>7}  {'支持':>6}  {'反对':>6}  {'中立':>6}   {'支持%':>6} {'反对%':>6} {'中立%':>6}"
    print(hdr)
    for dom, tot, s, a, n in rows:
        print(f"  {dom:<10}{tot:>7,}  {s:>6,}  {a:>6,}  {n:>6,}   {s/tot*100:>5.1f}% {a/tot*100:>5.1f}% {n/tot*100:>5.1f}%")
    # also report the neutral share (often the dominant class) to expose shift
    neutral_shares = {dom: n/tot for dom, tot, s, a, n in rows}
    print(f"\n  neutral-share range across domains: {min(neutral_shares.values())*100:.1f}% ~ "
          f"{max(neutral_shares.values())*100:.1f}%  (max gap {max(neutral_shares.values())-min(neutral_shares.values()):.3f})")

    # ------------------------------------------------------------------ #
    section(6, "SUBTASK B — cross-domain target & vocabulary overlap (DG)")
    # ------------------------------------------------------------------ #
    dom_test_targets = {}
    dom_test_texts = {}
    for folder, dom in DOMAINS.items():
        te = load(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        dom_test_targets[dom] = set(te["Target 1"])
        dom_test_texts[dom] = te["Text"].tolist()

    print("\nDistinct test targets per domain:")
    for dom, t in dom_test_targets.items():
        print(f"    {dom:<10} {len(t):,}")

    print("\nPairwise test-target overlap between domains (count / smaller-set %):")
    doms = list(DOMAINS.values())
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = doms[i], doms[j]
            inter = len(dom_test_targets[a] & dom_test_targets[b])
            if inter:
                denom = min(len(dom_test_targets[a]), len(dom_test_targets[b]))
                print(f"    {a} x {b}: {inter} ({inter/denom*100:.2f}%)")

    # vocabulary (character bigram) overlap between domains
    def vocab(texts, topn=5000):
        c = Counter()
        for t in texts:
            for i in range(len(t) - 1):
                c[t[i:i + 2]] += 1
        return set(w for w, _ in c.most_common(topn))

    print("\nCharacter-bigram vocabulary overlap (top-5000 per domain, test texts):")
    dom_vocab = {d: vocab(dom_test_texts[d]) for d in doms}
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = doms[i], doms[j]
            denom = min(len(dom_vocab[a]), len(dom_vocab[b]))
            inter = len(dom_vocab[a] & dom_vocab[b])
            print(f"    {a} x {b}: {inter/denom*100:.1f}% overlap")

    print("\nDone. (Charts can be generated next if wanted.)")


if __name__ == "__main__":
    main()
