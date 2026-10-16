# -*- coding: utf-8 -*-
"""训练循环：L_total = λc·L_CE + λs·L_SupCon + λp·L_Proto + λd·L_Domain。

- 域对抗（DANN）按 ganin 调度逐渐增大梯度反转系数。
- EDDA 表征扰动为 TODO（训练时内联）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.eval.metrics import macro_f1
from src.models.dann import ganin_lambda
from src.models.losses import CombinedLoss


class Trainer:
    def __init__(self, model, weights: dict, dg_cfg: dict, training_cfg: dict,
                 device: str = "cpu", log_dir: Optional[Path] = None):
        self.model = model.to(device)
        self.device = device
        self.training_cfg = training_cfg
        self.dg_cfg = dg_cfg
        self.log_dir = log_dir or Path("outputs/logs")

        no_decay = {"bias", "LayerNorm.weight"}
        grouped = [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)],
             "weight_decay": training_cfg.get("weight_decay", 0.01)},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(grouped, lr=training_cfg["learning_rate"])
        self.loss_fn = CombinedLoss(weights,
                                    supcon_t=dg_cfg.get("supcon", {}).get("temperature", 0.1),
                                    proto_t=dg_cfg.get("proto", {}).get("temperature", 0.1))

    def _to(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, scheduler=None, total_steps: int = 0):
        global_step = 0
        total = max(1, total_steps)
        best_f1, best_state = -1.0, None

        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                batch = self._to(batch)
                logits, feats = self.model(batch["input_ids"], batch["attention_mask"],
                                           return_feats=True)

                domain_logits, domain_ids = None, None
                if self.model.domain_disc is not None and "domain_id" in batch:
                    p = global_step / total
                    lam = ganin_lambda(p, self.dg_cfg["domain"].get("alpha", 10.0)) \
                        if self.dg_cfg["domain"].get("lambda_schedule") == "ganin" \
                        else self.dg_cfg["domain"].get("grl_lambda", 1.0)
                    self.model.grl.set_lambda(lam)
                    domain_logits = self.model.forward_domain(feats)
                    domain_ids = batch["domain_id"]

                # TODO(EDDA): 在 feats 上加对抗扰动作为表征级增强

                losses = self.loss_fn(
                    logits, feats, batch["label"],
                    target_ids=batch.get("target_id"),
                    domain_logits=domain_logits, domain_ids=domain_ids,
                    num_classes=self.model.classifier.out_features)

                self.optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.training_cfg.get("grad_clip", 1.0))
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            f1 = self.evaluate(val_loader)
            print(f"  [epoch {epoch+1}/{epochs}] val macro-F1 = {f1:.4f}")
            if f1 > best_f1:
                best_f1, best_state = f1, {k: v.detach().cpu().clone()
                                           for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_f1

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        preds, golds = [], []
        for batch in loader:
            batch = self._to(batch)
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            preds.append(logits.argmax(dim=-1).cpu())
            golds.append(batch["label"].cpu())
        return macro_f1(torch.cat(preds), torch.cat(golds))
