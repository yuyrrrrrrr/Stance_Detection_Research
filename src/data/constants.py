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
LABEL_TO_ID: dict = {"支持": 0, "反对": 1, "中立": 2}
ID_TO_LABEL: dict = {v: k for k, v in LABEL_TO_ID.items()}

# 目标类型：clauses 即论文中的 claim
TYPE_MAP: dict = {"noun_phrases": "noun_phrases", "clauses": "claim", "claims": "claim"}

# 子任务各划分的文件名
SPLIT_FILES: dict = {
    "train": "raw_train_all_onecol.csv",
    "val": "raw_val_all_onecol.csv",
    "test": "raw_test_all_onecol.csv",
}

# 增强风格标签（与真实域一起送入域判别器）
AUG_STYLES: list = ["orig", "eda", "backtranslation", "llm", "edda"]
