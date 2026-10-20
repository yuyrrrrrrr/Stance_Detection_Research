# -*- coding: utf-8 -*-
"""数据相关常量：领域文件夹映射、子任务文件命名。"""
from __future__ import annotations

# 领域文件夹名 -> 中文领域名（顺序固定，用于 domain_id 与配色）
DOMAIN_FOLDERS: dict = {
    "新冠疫情_covid_epidemic": "新冠疫情",
    "世界事件_world_event": "世界事件",
    "文化教育_culture_and_education": "文化教育",
    "娱乐和消费_entertainment_and_consumption": "娱乐和消费",
    "体育_sports": "体育",
    "权益_rights": "权益",
    "环保_enviromental_protection": "环保",  # 官方文件夹拼写即 enviromental
}
DOMAIN_ORDER: list = list(DOMAIN_FOLDERS.values())
DOMAIN_TO_ID: dict = {d: i for i, d in enumerate(DOMAIN_ORDER)}

# 标签：中文 -> id（0=支持 1=反对 2=中立）
# 注：立场标签（支持/反对/中立 → 0/1/2）映射的唯一来源是 configs/base.yaml，
# 经 src/common/config.py 的 Config.label_to_id / id_to_label 提供，此处不再重复，
# 避免出现第二份标签映射。

# 目标类型：clauses 即论文中的 claim；内部统一为二值 noun_phrase / claim
TYPE_MAP: dict = {"noun_phrases": "noun_phrase", "noun_phrase": "noun_phrase",
                  "clauses": "claim", "claims": "claim", "claim": "claim"}

# 子任务各划分的文件名
SPLIT_FILES: dict = {
    "train": "raw_train_all_onecol.csv",
    "val": "raw_val_all_onecol.csv",
    "test": "raw_test_all_onecol.csv",
}

# 增强风格标签（与真实域一起送入域判别器）
AUG_STYLES: list = ["orig", "eda", "backtranslation", "llm", "edda"]
