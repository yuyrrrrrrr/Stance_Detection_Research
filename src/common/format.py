# -*- coding: utf-8 -*-
"""输入模板的唯一来源。

训练与推理都必须通过 build_input 构造输入，保证线上推理与训练使用同一模板。
模板形如 "{target} [SEP] {text}"，经 tokenizer(add_special_tokens=True) 后为
    [CLS] {target} [SEP] {text} [SEP]
与 hfl/chinese-roberta-wwm-ext、bert-base-chinese 的 [SEP] 特殊标记一致。
"""
from __future__ import annotations

from typing import Optional

# 默认模板：目标在前、文本在后（立场检测常见做法）
DEFAULT_TEMPLATE = "{target} [SEP] {text}"


def build_input(text: str, target: str, template: Optional[str] = None,
                max_length: int = 128) -> str:
    """构造模型输入字符串（截断前，先拼接）。用 replace 而非 format，避免文本中的花括号干扰。"""
    template = template or DEFAULT_TEMPLATE
    s = (template.replace("{text}", text).replace("{target}", target)).strip()
    # 长度截断交给 tokenizer（truncation=True + max_length），这里只返回模板化结果
    return s
