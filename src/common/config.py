# -*- coding: utf-8 -*-
"""配置加载：读取 configs/*.yaml，合并 base/da/dg，并把相对路径解析为绝对路径。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# 项目根目录：src/common/config.py -> 上溯两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"缺少配置文件：{path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    base: Dict[str, Any]
    da: Dict[str, Any] = field(default_factory=dict)
    dg: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, base: str = "base.yaml", da: str = "da.yaml",
             dg: str = "dg.yaml") -> "Config":
        return cls(load_yaml(base), load_yaml(da), load_yaml(dg))

    # ---- 路径解析 ----
    def path(self, key: str) -> Path:
        p = self.base.get("paths", {}).get(key)
        if p is None:
            raise KeyError(f"configs/base.yaml 的 paths 缺少键 {key!r}")
        p = Path(p)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def raw_root(self) -> Path:
        return self.path("raw_root")

    # ---- 标签映射 ----
    @property
    def labels(self) -> List[str]:
        return self.base["labels"]

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {lbl: i for i, lbl in enumerate(self.labels)}

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {i: lbl for i, lbl in enumerate(self.labels)}

    # ---- 便捷取值 ----
    @property
    def input_template(self) -> str:
        return self.base["input_template"]["pattern"]

    @property
    def max_length(self) -> int:
        return self.base["input_template"]["max_length"]

    @property
    def backbone(self) -> str:
        return self.base["model"]["backbone"]

    @property
    def num_labels(self) -> int:
        return self.base["num_labels"]
