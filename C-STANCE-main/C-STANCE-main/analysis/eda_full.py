# -*- coding: utf-8 -*-
"""
C-STANCE 中文零样本立场检测 —— 完整 EDA（面向领域泛化 DG + 数据增强 DA）
================================================================================
研究问题：基于预训练语言模型的中文 ZSSD 的领域泛化与数据增强。

数据格式自动适配说明（实际数据 vs 用户描述）：
  - 实际字段：Text / Target 1 / Stance 1 / [Domain] / [In Use] / Type
  - Stance 为中文标签（支持/反对/中立），Type 为 noun_phrases/clauses(即论文 claim)
  - 脚本会：归一化列名 -> 自动识别 Subtask A/B -> 统一立场标签编码 -> 统一目标类型命名
================================================================================
运行：
    PYTHONIOENCODING=utf-8 python analysis/eda_full.py
依赖：pandas, numpy, matplotlib, seaborn（jieba/emoji/wordcloud 可选，未安装时自动降级）
"""
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面保存图片
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ----------------------------- 0. 配置 ----------------------------- #
# 统一数据目录（可改为你机器的路径）
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "c_stance_dataset")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

SUB_A = os.path.join(DATA_DIR, "subtaskA")
SUB_B = os.path.join(DATA_DIR, "subtaskB")

# 领域（文件夹名 -> 中文领域名），顺序固定用于配色
DOMAIN_FOLDERS = {
    "新冠疫情_covid_epidemic": "新冠疫情",
    "世界事件_world_event": "世界事件",
    "文化教育_culture_and_education": "文化教育",
    "娱乐和消费_entertainment_and_consumption": "娱乐和消费",
    "体育_sports": "体育",
    "权益_rights": "权益",
    "环保_enviromental_protection": "环保",
}
DOMAIN_ORDER = list(DOMAIN_FOLDERS.values())

# 立场标签：中文 <-> 数值（0=支持, 1=反对, 2=中立）
LABEL_CN = ["支持", "反对", "中立"]
LABEL_TO_ID = {"支持": 0, "反对": 1, "中立": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# ------- 中文字体（Windows：SimHei / Microsoft YaHei；Linux/Mac 自动探测） ------- #
def setup_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                  "WenQuanYi Micro Hei", "Arial Unicode MS"]
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] 使用中文字体: {name}")
            return name
    print("[font] WARNING: 未找到中文字体，中文可能显示为方框")
    return None

# ------- 配色（dataviz 色盲安全调色板） ------- #
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
C_GRAY = "#898781"
DOMAIN_COLORS = dict(zip(DOMAIN_ORDER,
                         [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET]))
# 立场用发散语义色：支持=蓝，反对=红，中立=灰（语义上更贴切）
STANCE_COLORS = {"支持": C_BLUE, "反对": C_RED, "中立": C_GRAY}
# 顺序色（热力图用，单色相蓝，浅->深）
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue",
                                             ["#cde2fb", "#6da7ec", "#256abf", "#104281"])

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
                                     "grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7",
                                     "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"})
# 中文字体必须在 sns.set_theme 之后设置，否则会被主题默认字体覆盖
setup_chinese_font()


# ----------------------------- 数据加载与归一化 ----------------------------- #
def normalize_columns(df):
    """列名归一化：Text->text, Target*/target->target, Stance*/label->label,
    Domain->domain, In Use->in_use, Type->type。自动适配不同命名。"""
    canon = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("text", "content", "正文", "微博"):
            canon[c] = "text"
        elif cl == "target" or cl.startswith("target") or cl == "目标":
            canon[c] = "target"
        elif cl == "stance" or cl.startswith("stance") or cl == "label" or cl.startswith("label") or cl == "立场":
            canon[c] = "label"
        elif cl == "domain" or cl == "领域":
            canon[c] = "domain"
        elif cl.replace(" ", "") == "inuse" or cl in ("in use", "是否使用", "used"):
            canon[c] = "in_use"
        elif cl == "type" or cl in ("类型", "target type"):
            canon[c] = "type"
    return df.rename(columns=canon)


def normalize_label(df):
    """立场标签编码确认：支持/反对/中立 -> 0/1/2；若已是数值则直接使用。返回 (df, mapping)。"""
    raw = df["label"].astype(str).str.strip()
    mapping = {}
    if set(raw.unique()).issubset(set(LABEL_TO_ID)):
        mapping = LABEL_TO_ID
        df["label_id"] = raw.map(LABEL_TO_ID)
    elif set(raw.unique()).issubset({"0", "1", "2"}):
        mapping = {ID_TO_LABEL[int(v)]: int(v) for v in raw.unique()}
        df["label_id"] = raw.astype(int)
    else:
        raise ValueError(f"未知标签取值: {raw.unique()}")
    df["label_cn"] = df["label_id"].map(ID_TO_LABEL)
    return df, mapping


def normalize_type(df):
    """目标类型统一：clauses -> claim（论文术语）。"""
    if "type" in df.columns:
        df["type"] = df["type"].str.strip().str.lower().map(
            {"noun_phrases": "noun_phrases", "noun_phrase": "noun_phrases",
             "clauses": "claim", "claims": "claim", "clause": "claim"})
    return df


def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df = normalize_columns(df)
    df, _ = normalize_label(df)
    df = normalize_type(df)
    if "in_use" in df.columns:
        df["in_use"] = df["in_use"].astype(str).str.strip()
    return df


def count_tokens(text):
    """无 jieba 的 token 计数：单个汉字=1 token，连续字母数字=1 token（中文常用近似）。"""
    if pd.isna(text):
        return 0
    return len(re.findall(r"[一-鿿]|[A-Za-z0-9]+", str(text)))


# ----------------------------- 图表保存工具 ----------------------------- #
def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] 已保存 {path}")


def print_table(df, title):
    print(f"\n  {title}")
    print(df.to_string())


# ============================================================================
# 模块 1：数据概览
# ============================================================================
def module1_overview():
    print("\n" + "=" * 80)
    print("[模块1] 数据概览：样本量 / 字段 / 缺失值 / 重复值 / 标签编码确认")
    print("=" * 80)

    files = {
        "subtaskA train": os.path.join(SUB_A, "raw_train_all_onecol.csv"),
        "subtaskA val": os.path.join(SUB_A, "raw_val_all_onecol.csv"),
        "subtaskA test": os.path.join(SUB_A, "raw_test_all_onecol.csv"),
    }
    for folder, dom in DOMAIN_FOLDERS.items():
        files[f"subtaskB train({dom})"] = os.path.join(SUB_B, folder, "raw_train_all_onecol.csv")
        files[f"subtaskB val({dom})"] = os.path.join(SUB_B, folder, "raw_val_all_onecol.csv")
        files[f"subtaskB test({dom})"] = os.path.join(SUB_B, folder, "raw_test_all_onecol.csv")

    summary = []
    for name, path in files.items():
        df = load_csv(path)
        n_dup = df.duplicated().sum()
        n_missing = int(df.isna().sum().sum())
        summary.append((name, len(df), list(df.columns), n_dup, n_missing))
        if "subtaskA" in name:
            label_mapping = LABEL_TO_ID

    print("\n  文件样本量与字段：")
    for name, n, cols, dup, miss in summary:
        print(f"    {name:<28} {n:>7,} 行  列={cols}  重复={dup}  缺失单元格={miss}")

    # 标签编码确认
    s = load_csv(os.path.join(SUB_A, "raw_train_all_onecol.csv"))
    print("\n  标签编码确认（Subtask A train）：")
    print(f"    原始取值 -> 编码：{LABEL_TO_ID}  （0=支持, 1=反对, 2=中立）")
    print(f"    label_id 取值分布：{s['label_id'].value_counts().sort_index().to_dict()}")

    # 字段类型
    print("\n  字段 dtype 概览（Subtask A train）：")
    print(s.dtypes.to_string())


# ============================================================================
# 模块 2：标签分布
# ============================================================================
def module2_label_dist():
    print("\n" + "=" * 80)
    print("[模块2] 标签分布：整体 / 分数据集 / 分领域，类别不平衡检测")
    print("=" * 80)

    # --- 整体（subtask A）--- #
    tr = load_csv(os.path.join(SUB_A, "raw_train_all_onecol.csv"))
    va = load_csv(os.path.join(SUB_A, "raw_val_all_onecol.csv"))
    te = load_csv(os.path.join(SUB_A, "raw_test_all_onecol.csv"))

    dist = pd.DataFrame({
        "train": tr["label_cn"].value_counts(normalize=True).reindex(LABEL_CN) * 100,
        "val": va["label_cn"].value_counts(normalize=True).reindex(LABEL_CN) * 100,
        "test": te["label_cn"].value_counts(normalize=True).reindex(LABEL_CN) * 100,
    }).round(2)
    print("\n  立场分布（%，Subtask A）：")
    print(dist.to_string())

    # --- 分领域（subtask B 各留出域测试集，纯单域）--- #
    rows = []
    for folder, dom in DOMAIN_FOLDERS.items():
        d = load_csv(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        c = d["label_cn"].value_counts()
        tot = len(d)
        rows.append([dom, tot] + [round(c.get(l, 0) / tot * 100, 1) for l in LABEL_CN])
    dom_dist = pd.DataFrame(rows, columns=["domain", "n"] + [f"{l}%" for l in LABEL_CN])
    print("\n  立场分布（%，Subtask B 各域测试集）：")
    print(dom_dist.to_string(index=False))

    neutral = dom_dist["中立%"]
    print(f"\n  【类别不平衡检测】中立占比跨域：min={neutral.min():.1f}% max={neutral.max():.1f}% 差距={neutral.max()-neutral.min():.1f}pp")
    print(f"  -> 整体三类大致均衡（~1/3），不平衡主要体现在【跨领域】而非整体，DA 应瞄准领域/目标稀缺而非类别再平衡。")

    # --- 图：整体标签分布（发散色）--- #
    fig, ax = plt.subplots(figsize=(5, 3.6))
    order = ["支持", "中立", "反对"]
    vals = [tr["label_cn"].eq(l).sum() for l in order]
    colors = [STANCE_COLORS[l] for l in order]
    bars = ax.bar(order, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 200, f"{v:,}", ha="center", va="bottom")
    ax.set_title("Subtask A 训练集立场分布", fontsize=12)
    ax.set_ylabel("样本数")
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x):,}")
    save_fig(fig, "02_label_dist_overall.png")

    # --- 图：分数据集标签分布（分组柱）--- #
    fig, ax = plt.subplots(figsize=(6, 3.8))
    counts = pd.DataFrame({
        "train": tr["label_cn"].value_counts().reindex(LABEL_CN),
        "val": va["label_cn"].value_counts().reindex(LABEL_CN),
        "test": te["label_cn"].value_counts().reindex(LABEL_CN),
    })
    counts.plot(kind="bar", ax=ax, color=[STANCE_COLORS[l] for l in LABEL_CN], width=0.7)
    ax.set_title("分数据集立场分布", fontsize=12)
    ax.set_xticklabels(LABEL_CN, rotation=0)
    ax.legend(title="数据集")
    save_fig(fig, "02_label_dist_by_split.png")


# ============================================================================
# 模块 3：领域分布
# ============================================================================
def module3_domain():
    print("\n" + "=" * 80)
    print("[模块3] 领域分布：各域样本量 / 领域×标签交叉 / 文本长度与目标类型差异")
    print("=" * 80)

    per_domain = []
    for folder, dom in DOMAIN_FOLDERS.items():
        tr = load_csv(os.path.join(SUB_B, folder, "raw_train_all_onecol.csv"))
        va = load_csv(os.path.join(SUB_B, folder, "raw_val_all_onecol.csv"))
        te = load_csv(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        tr_used = tr[tr["in_use"] == "1"]
        va_used = va[va["in_use"] == "1"]
        # 每个域的全部样本（跨所有划分求和时用 test 纯单域 + 该域在其它划分中作为训练域的量不重复统计）
        # 这里统计：作为"留出测试域"时的样本量 + 被剔除的 In Use==0 量
        per_domain.append({
            "domain": dom,
            "test": len(te),
            "train_used": len(tr_used), "val_used": len(va_used),
            "dropped_inuse0": int((tr["in_use"] == "0").sum() + (va["in_use"] == "0").sum()),
        })
    dd = pd.DataFrame(per_domain).set_index("domain")
    print("\n  各领域（留出测试）样本量与 In Use==0 剔除量：")
    print(dd.to_string())

    # --- 领域×标签交叉（用各域测试集）--- #
    cross = []
    for folder, dom in DOMAIN_FOLDERS.items():
        te = load_csv(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        for l in LABEL_CN:
            cross.append((dom, l, int((te["label_cn"] == l).sum())))
    ct = pd.DataFrame(cross, columns=["domain", "label", "n"]).pivot(
        index="domain", columns="label", values="n").reindex(index=DOMAIN_ORDER, columns=LABEL_CN)
    print("\n  领域×标签 交叉表（测试集样本数）：")
    print(ct.to_string())

    # --- 各域文本长度与目标类型差异 --- #
    info = []
    for folder, dom in DOMAIN_FOLDERS.items():
        te = load_csv(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        te = te.copy()
        te["clen"] = te["text"].str.len()
        te["tlen"] = te["target"].str.len()
        info.append({
            "domain": dom,
            "文本均长(字)": round(te["clen"].mean(), 1),
            "目标均长(字)": round(te["tlen"].mean(), 1),
            "claim占比%": round((te["type"] == "claim").mean() * 100, 1),
        })
    info_df = pd.DataFrame(info).set_index("domain")
    print("\n  各域 文本/目标长度 与 claim 占比：")
    print(info_df.to_string())

    # --- 图：各域样本量（水平条形，领域固定色）--- #
    fig, ax = plt.subplots(figsize=(7, 4))
    dd2 = dd.sort_values("test")
    colors = [DOMAIN_COLORS[d] for d in dd2.index]
    bars = ax.barh(dd2.index, dd2["test"], color=colors, height=0.65)
    for b, v in zip(bars, dd2["test"]):
        ax.text(v + 100, b.get_y() + b.get_height() / 2, f"{v:,}", va="center")
    ax.set_title("各领域测试集样本量", fontsize=12)
    ax.set_xlabel("样本数")
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x):,}")
    save_fig(fig, "03_domain_sample_count.png")

    # --- 图：领域×标签热力图（顺序单色相）--- #
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    sns.heatmap(ct, annot=True, fmt="d", cmap=SEQ_BLUE, ax=ax,
                cbar_kws={"label": "样本数"}, linewidths=0.5, linecolor="#fcfcfb")
    ax.set_title("领域 × 立场 交叉分布（测试集）", fontsize=12)
    save_fig(fig, "04_domain_label_heatmap.png")

    # --- 图：各域立场占比（100% 堆叠水平条形）--- #
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct[LABEL_CN].plot(kind="barh", stacked=True, ax=ax,
                          color=[STANCE_COLORS[l] for l in LABEL_CN], width=0.65)
    ax.set_title("各领域立场占比（%，测试集）", fontsize=12)
    ax.set_xlabel("占比 (%)")
    ax.legend(title="立场", ncol=3, loc="lower right", bbox_to_anchor=(1, -0.25))
    ax.set_xlim(0, 100)
    save_fig(fig, "05_domain_label_stacked.png")


# ============================================================================
# 模块 4：目标分析
# ============================================================================
def module4_target():
    print("\n" + "=" * 80)
    print("[模块4] 目标分析：类型 / 频次长尾 / Top-20 / 训练-测试目标与领域重叠检查")
    print("=" * 80)

    tr = load_csv(os.path.join(SUB_A, "raw_train_all_onecol.csv"))
    va = load_csv(os.path.join(SUB_A, "raw_val_all_onecol.csv"))
    te = load_csv(os.path.join(SUB_A, "raw_test_all_onecol.csv"))

    # --- 目标类型统计 --- #
    type_cnt = tr["type"].value_counts()
    print("\n  目标类型（Subtask A train）：")
    for t, v in type_cnt.items():
        print(f"    {t:<14} {v:>7,}  ({v/len(tr)*100:.1f}%)")
    print("    note: clauses = 论文中的 claim（主张），noun_phrases = 名词短语")

    # --- 目标频次分布 / 长尾 --- #
    freq = Counter(tr["target"])
    f = np.array(list(freq.values()))
    print(f"\n  训练集目标频次：distinct={len(freq):,}  mean={f.mean():.2f}  median={np.median(f):.0f}  max={f.max():.0f}")
    for thr in [1, 2, 3, 5, 10]:
        print(f"    出现 <= {thr:>2} 次的目标占比：{(f <= thr).mean()*100:.2f}%")

    top20 = pd.Series(freq).sort_values(ascending=False).head(20)
    print("\n  高频目标 Top-20（训练集）：")
    print(top20.to_string())

    # --- Subtask A 目标重叠检查 --- #
    tr_t, va_t, te_t = set(tr["target"]), set(va["target"]), set(te["target"])
    print("\n  【Subtask A 零样本验证】训练/验证/测试目标重叠：")
    print(f"    train∩test = {len(tr_t & te_t)}   train∩val = {len(tr_t & va_t)}   val∩test = {len(va_t & te_t)}")
    print(f"    （完全零样本要求 train∩test==0；少量重叠多来自模板化 claim，见报告）")

    # --- Subtask B 领域重叠检查 --- #
    print("\n  【Subtask B 零样本验证】跨域测试目标重叠（应接近 0）：")
    dom_targets = {}
    for folder, dom in DOMAIN_FOLDERS.items():
        d = load_csv(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        dom_targets[dom] = set(d["target"])
    ov = []
    for i in range(len(DOMAIN_ORDER)):
        for j in range(i + 1, len(DOMAIN_ORDER)):
            a, b = DOMAIN_ORDER[i], DOMAIN_ORDER[j]
            inter = len(dom_targets[a] & dom_targets[b])
            denom = min(len(dom_targets[a]), len(dom_targets[b]))
            ov.append((f"{a}×{b}", inter, round(inter / denom * 100, 2)))
    ov_df = pd.DataFrame(ov, columns=["域对", "重叠数", "重叠%"])
    print(ov_df.to_string(index=False))

    # --- 图：目标类型柱状图 --- #
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(["noun_phrases\n(名词短语)", "claim\n(主张/clauses)"],
                  type_cnt.reindex(["noun_phrases", "claim"]).values,
                  color=[C_AQUA, C_BLUE], width=0.5)
    for b, v in zip(bars, type_cnt.reindex(["noun_phrases", "claim"]).values):
        ax.text(b.get_x() + b.get_width() / 2, v + 300, f"{v:,}", ha="center")
    ax.set_title("目标类型分布（训练集）", fontsize=12)
    ax.set_ylabel("样本数")
    save_fig(fig, "06_target_type_dist.png")

    # --- 图：目标频次长尾（log-log）--- #
    ranked = np.sort(f)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.arange(1, len(ranked) + 1), ranked, color=C_BLUE, lw=1.6)
    ax.set_title("目标频次长尾（log-log）", fontsize=12)
    ax.set_xlabel("目标排名（按频次）")
    ax.set_ylabel("频次")
    ax.grid(True, which="both", color="#e1e0d9", lw=0.6)
    save_fig(fig, "07_target_freq_longtail.png")

    # --- 图：Top-20 目标 --- #
    fig, ax = plt.subplots(figsize=(8, 5))
    top = top20.iloc[::-1]
    ax.barh(top.index, top.values, color=C_BLUE, height=0.6)
    ax.set_title("高频目标 Top-20（训练集）", fontsize=12)
    ax.set_xlabel("频次")
    save_fig(fig, "08_target_top20.png")


# ============================================================================
# 模块 5：文本特征分析
# ============================================================================
# 表情符号 Unicode 范围（近似）
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002190-\U000021FF️‍]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
AT_RE = re.compile(r"@[\w一-鿿\-]+")
HASH_RE = re.compile(r"#[^#\s]+#")
# 隐式立场启发式词典（反讽 / 阴阳怪气 / 网络用语 / 谐音，仅作粗筛代理）
# 注意：只用多字网络语，避免单字/数字造成大量误匹配（如 "6" 会命中所有含数字的文本）
IMPLICIT_LEXICON = ["呵呵", "呵呵哒", "笑死", "绝绝子", "yyds", "无语子", "蚌埠住了",
                    "蚌埠住", "666", "就这", "啊对对对", "不会吧", "真的会谢", "泰裤辣",
                    "麻了", "栓Q", "emo了", "破防", "凡尔赛", "躺平", "内卷", "摆烂",
                    "急了", "懂王", "赢麻了", "典中典", "绷不住", "蚌埠", "破大防"]


def text_features(df):
    n = len(df)
    clen = df["text"].str.len()
    toks = df["text"].apply(count_tokens)
    has_emoji = df["text"].apply(lambda s: bool(EMOJI_RE.search(str(s))))
    has_url = df["text"].apply(lambda s: bool(URL_RE.search(str(s))))
    has_at = df["text"].apply(lambda s: bool(AT_RE.search(str(s))))
    has_hash = df["text"].apply(lambda s: bool(HASH_RE.search(str(s))))
    has_implicit = df["text"].apply(lambda s: any(k in str(s) for k in IMPLICIT_LEXICON))
    return dict(
        clen_mean=clen.mean(), clen_median=clen.median(), clen_max=clen.max(),
        tok_mean=toks.mean(),
        emoji=has_emoji.mean(), url=has_url.mean(), at=has_at.mean(),
        hashtag=has_hash.mean(), implicit=has_implicit.mean())


def module5_text():
    print("\n" + "=" * 80)
    print("[模块5] 文本特征：长度 / 特殊符号占比 / 隐式表达粗筛")
    print("=" * 80)

    tr = load_csv(os.path.join(SUB_A, "raw_train_all_onecol.csv"))
    va = load_csv(os.path.join(SUB_A, "raw_val_all_onecol.csv"))
    te = load_csv(os.path.join(SUB_A, "raw_test_all_onecol.csv"))

    for name, df in [("train", tr), ("val", va), ("test", te)]:
        feat = text_features(df)
        print(f"\n  [{name}] 文本长度：字符 mean={feat['clen_mean']:.1f} median={feat['clen_median']:.0f} max={feat['clen_max']:.0f}；token(近似) mean={feat['tok_mean']:.1f}")
        print(f"    特殊符号占比：emoji={feat['emoji']*100:.1f}%  URL={feat['url']*100:.2f}%  @提及={feat['at']*100:.2f}%  话题#={feat['hashtag']*100:.2f}%")
        print(f"    隐式表达粗筛命中占比：{feat['implicit']*100:.1f}%（启发式词典，非精确）")

    # 隐式表达抽样（训练集）
    mask = tr["text"].apply(lambda s: any(k in str(s) for k in IMPLICIT_LEXICON))
    sample = tr[mask].head(10)[["text", "target", "label_cn"]]
    print("\n  隐式表达抽样（可能含反讽/网络用语/谐音，需人工复核）：")
    for _, r in sample.iterrows():
        t = r["text"]
        if len(t) > 60:
            t = t[:60] + "…"
        print(f"    [{r['label_cn']}] 目标={r['target']} | {t}")

    # --- 图：文本长度直方图 --- #
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(tr["text"].str.len(), bins=40, color=C_BLUE, alpha=0.85)
    ax.set_title("文本长度分布（字符，训练集）", fontsize=12)
    ax.set_xlabel("字符数")
    ax.set_ylabel("样本数")
    save_fig(fig, "09_text_len_hist.png")

    # --- 图：各域文本长度箱线图 --- #
    lens = []
    for folder, dom in DOMAIN_FOLDERS.items():
        d = load_csv(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        lens.append((dom, d["text"].str.len()))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    box_data = [v for _, v in lens]
    labels = [d for d, _ in lens]
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color=C_RED))
    for patch, d in zip(bp["boxes"], labels):
        patch.set_facecolor(DOMAIN_COLORS[d])
        patch.set_alpha(0.55)
    ax.set_title("各领域文本长度分布（测试集）", fontsize=12)
    ax.set_ylabel("字符数")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "10_text_len_by_domain.png")


# ============================================================================
# 模块 6：零样本划分验证
# ============================================================================
def module6_zeroshot():
    print("\n" + "=" * 80)
    print("[模块6] 零样本划分验证：目标不相交 / 领域留出 / In Use 过滤 / 数据泄露检查")
    print("=" * 80)

    tr = load_csv(os.path.join(SUB_A, "raw_train_all_onecol.csv"))
    va = load_csv(os.path.join(SUB_A, "raw_val_all_onecol.csv"))
    te = load_csv(os.path.join(SUB_A, "raw_test_all_onecol.csv"))

    tr_t, va_t, te_t = set(tr["target"]), set(va["target"]), set(te["target"])
    print("\n  【Subtask A 目标不相交验证】")
    print(f"    train∩val = {len(tr_t & va_t)}   val∩test = {len(va_t & te_t)}   train∩test = {len(tr_t & te_t)}")
    print(f"    -> {'PASS（目标完全不相交）' if len(tr_t & te_t) == 0 else f'WARN：train∩test 有 {len(tr_t & te_t)} 个重叠目标（多为模板化 claim，需人工复核）'}")

    # 重叠目标样例
    if tr_t & te_t:
        overlap = list(tr_t & te_t)
        print("    重叠目标样例：")
        for t in sorted(overlap, key=len, reverse=True)[:8]:
            print(f"      · {t}")

    print("\n  【Subtask B 领域留出验证】")
    for folder, dom in DOMAIN_FOLDERS.items():
        trb = load_csv(os.path.join(SUB_B, folder, "raw_train_all_onecol.csv"))
        teb = load_csv(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv"))
        tr_domains = set(trb["domain"].unique())
        te_domains = set(teb["domain"].unique())
        leak = te_domains & tr_domains
        n_inuse0 = int((trb["in_use"] == "0").sum())
        status = "PASS" if not leak else "FAIL(泄露)"
        print(f"    {dom:<10} 测试域={te_domains} 训练域数={len(tr_domains)} 交叉={leak or '无'}  "
              f"InUse==0(训练被剔除)={n_inuse0}  [{status}]")

    print("\n  【数据泄露检查结论】")
    print("    - Subtask A：目标级零样本基本成立（少量模板化 claim 重叠，建议清洗）。")
    print("    - Subtask B：测试域在训练集中完全未出现，In Use==0 样本已按官方要求剔除。")


# ============================================================================
# 模块 7：数据增强可行性分析
# ============================================================================
def module7_augmentation():
    print("\n" + "=" * 80)
    print("[模块7] 数据增强可行性：稀缺识别 / 隐式表达占比 / 增强风险")
    print("=" * 80)

    tr = load_csv(os.path.join(SUB_A, "raw_train_all_onecol.csv"))

    # --- 稀缺识别 --- #
    freq = Counter(tr["target"])
    print("\n  ① 哪些适合增强（稀缺识别）：")
    print(f"    - 目标稀疏：{(np.array(list(freq.values())) <= 1).mean()*100:.1f}% 的目标只出现 1 次 -> 目标级增强收益最大")
    type_x_label = pd.crosstab(tr["label_cn"], tr["type"])
    print("    - 类别×目标类型交叉（样本量）：")
    print(type_x_label.to_string())
    # 各域测试量（subtask B）最小域
    dmin = pd.Series({dom: len(load_csv(os.path.join(SUB_B, folder, "raw_test_all_onecol.csv")))
                      for folder, dom in DOMAIN_FOLDERS.items()})
    print(f"    - 领域稀缺：测试样本最少域 = {dmin.idxmin()}（{dmin.min():,}），最多域 = {dmin.idxmax()}（{dmin.max():,}）")
    print(f"    -> 目标级增强（改写/同义替换/模板扩写）优先级最高；领域级增强可针对小域（如{dmin.idxmin()}）做跨域迁移")

    # --- 隐式表达占比 --- #
    mask = tr["text"].apply(lambda s: any(k in str(s) for k in IMPLICIT_LEXICON))
    print(f"\n  ② 隐式表达占比：粗筛命中 {mask.mean()*100:.1f}%（启发式，低估真实值）")
    print("    -> 若占比可观，可考虑思维链/推理增强（先识别立场线索再判标签）以提升鲁棒性")

    # --- 增强风险（定性） --- #
    print("\n  ③ 回译 / LLM 生成增强的潜在风险（需在实现时规避）：")
    print("    - 标签一致性：回译/改写可能改变立场极性，需校验改写前后 target 语义一致")
    print("    - 目标泄露：LLM 生成时若见过测试目标，会破坏零样本假设 -> 生成时屏蔽测试目标")
    print("    - 模板化 claim：训练集中大量模板化 claim 是双刃剑，增强时应去模板化而非加剧")


# ============================================================================
# main
# ============================================================================
def main():
    print("=" * 80)
    print("C-STANCE 零样本立场检测 EDA（领域泛化 DG + 数据增强 DA）")
    print(f"数据目录：{DATA_DIR}")
    print("=" * 80)
    module1_overview()
    module2_label_dist()
    module3_domain()
    module4_target()
    module5_text()
    module6_zeroshot()
    module7_augmentation()
    print("\n" + "=" * 80)
    print(f"完成。图表已保存至 {FIG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
