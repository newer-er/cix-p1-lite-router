"""Inference-only architecture, state-dict compatible with the M4 training code.

This module is imported by the classifier process only. The gateway has no torch
dependency. Training remains unchanged; compatibility is checked against it.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from transformers import ModernBertConfig, ModernBertModel

from .classifier_contract import LABELS, MULTI_LABELS


class TaskAttentionPool(nn.Module):
    def __init__(self, hidden: int, tasks: int):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(tasks, 2, hidden))

    def forward(self, hidden, attention_mask, current_token_mask):
        full = attention_mask.bool()
        current = current_token_mask.bool() & full
        current = torch.where(current.any(-1, keepdim=True), current, full)
        keep = torch.stack((full, current), -1).unsqueeze(2)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            values = hidden.float()
            scores = torch.einsum("bth,dqh->btdq", values, self.queries.float()) / math.sqrt(hidden.shape[-1])
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=1) * keep
            probs = probs / probs.sum(1, keepdim=True).clamp(min=1e-12)
            pooled = torch.einsum("btdq,bth->bdqh", probs, values)
        return pooled.flatten(2).to(hidden.dtype)


class RouterClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config.get("labels") != LABELS or config.get("multi_labels") != MULTI_LABELS:
            raise ValueError("classifier_labels_differ_from_taxonomy")
        if config.get("residual_intent_head"):
            raise ValueError("classifier_residual_checkpoint_not_supported")
        opts = config["training_methods"]
        if opts["pooling"] != "attention_current":
            raise ValueError("classifier_pooling_not_supported")
        enc_config = ModernBertConfig.from_dict(config["encoder_config"])
        enc_config._attn_implementation = "sdpa"
        self.encoder = ModernBertModel(enc_config)
        hidden = enc_config.hidden_size
        self.pooling = opts["pooling"]
        self.dropout = nn.Dropout(opts["dropout"])
        self.heads = nn.ModuleDict({d: nn.Linear(hidden * 2, len(v)) for d, v in LABELS.items()})
        self.multi_heads = nn.ModuleDict({d: nn.Linear(hidden * 2, len(v)) for d, v in MULTI_LABELS.items()})
        self.attention_pool = TaskAttentionPool(hidden, len(LABELS) + len(MULTI_LABELS))

    def forward(self, input_ids, attention_mask, current_token_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pools = self.dropout(self.attention_pool(hidden, attention_mask, current_token_mask))
        return ({d: self.heads[d](pools[:, i]) for i, d in enumerate(LABELS)},
                {d: self.multi_heads[d](pools[:, len(LABELS) + i]) for i, d in enumerate(MULTI_LABELS)})


class QwenRouterClassifier(nn.Module):
    """M4.7 text backbone with the trained LoRA updates merged into FP32 weights."""

    def __init__(self, config: dict):
        super().__init__()
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        if config.get("labels") != LABELS or config.get("multi_labels") != MULTI_LABELS:
            raise ValueError("classifier_labels_differ_from_taxonomy")
        opts = config["training_methods"]
        if opts["pooling"] != "attention_current":
            raise ValueError("classifier_pooling_not_supported")
        enc_config = Qwen3_5TextConfig.from_dict(config["encoder_config"])
        enc_config._attn_implementation = "sdpa"
        # Classification is a single full forward pass; no generation cache is used.
        enc_config.use_cache = False
        self.encoder = Qwen3_5TextModel(enc_config)
        hidden = enc_config.hidden_size
        self.dropout = nn.Dropout(opts["dropout"])
        self.heads = nn.ModuleDict({d: nn.Linear(hidden * 2, len(v)) for d, v in LABELS.items()})
        self.multi_heads = nn.ModuleDict({d: nn.Linear(hidden * 2, len(v)) for d, v in MULTI_LABELS.items()})
        self.attention_pool = TaskAttentionPool(hidden, len(LABELS) + len(MULTI_LABELS))

    def forward(self, input_ids, attention_mask, current_token_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        pools = self.dropout(self.attention_pool(hidden, attention_mask, current_token_mask))
        return ({d: self.heads[d](pools[:, i]) for i, d in enumerate(LABELS)},
                {d: self.multi_heads[d](pools[:, len(LABELS) + i]) for i, d in enumerate(MULTI_LABELS)})


def build_classifier(config: dict):
    architecture = config.get("architecture")
    if architecture == "qwen35_08b_merged_attention_heads":
        return QwenRouterClassifier(config)
    if architecture not in (None, "mmbert_encoder_attention_current_pool_fixed_label_heads"):
        raise ValueError("classifier_architecture_not_supported")
    return RouterClassifier(config)
