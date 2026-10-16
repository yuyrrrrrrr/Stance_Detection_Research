# -*- coding: utf-8 -*-
"""FastAPI 三分类推理服务：text + target -> 立场（支持/反对/中立）与概率。

线上推理与训练共用同一输入模板（src.common.format.build_input），保证一致。
启动：uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
依赖：pip install -r requirements-serve.txt
"""
from __future__ import annotations

import os
from typing import Dict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError("serve 需要 fastapi/uvicorn，请先 `pip install -r requirements-serve.txt`") from e

from src.common.config import Config
from src.common.format import build_input
from src.models import StanceClassifier


class PredictRequest(BaseModel):
    text: str
    target: str


class PredictResponse(BaseModel):
    label: str
    label_id: int
    probs: Dict[str, float]


app = FastAPI(title="中文零样本立场检测 API", version="0.1.0")

_cfg = Config.load()
_model = None
_tokenizer = None


def load_model():
    """惰性加载模型与分词器（首个请求时加载，或由启动事件调用）。"""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer
        import torch
        ckpt_path = os.environ.get("CKPT_PATH", "outputs/ckpt/best.pt")
        _model = StanceClassifier(_cfg.backbone, _cfg.num_labels,
                                  dropout=_cfg.base["model"]["dropout"])
        if os.path.exists(ckpt_path):
            _model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
        _model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(_cfg.backbone)
    return _model, _tokenizer


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    import torch
    model, tokenizer = load_model()
    inp = build_input(req.text, req.target, _cfg.input_template, _cfg.max_length)
    enc = tokenizer(inp, truncation=True, max_length=_cfg.max_length,
                    return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    label_id = int(logits.argmax(dim=-1).item())
    return PredictResponse(
        label=_cfg.id_to_label[label_id],
        label_id=label_id,
        probs={_cfg.id_to_label[i]: round(p, 4) for i, p in enumerate(probs)},
    )
