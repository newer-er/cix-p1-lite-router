"""Explicit policy between request classification and existing Worker selection.

The classifier predicts labels, not Worker outcomes. This initial policy uses a
confident domain and optionally coding fix/debug intent. Capability predictions
remain diagnostic; actual image/length constraints stay in the existing router.
The threshold is an operational setting, not a measured quality guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

from .decision_backend import DecisionResult


@dataclass(frozen=True)
class ClassificationContext:
    task_type: str
    trace: dict


def classification_context(baseline: str, result: DecisionResult, *, mode: str, min_confidence: float = 0.7):
    trace = result.to_trace()
    trace.update({"mode": mode, "applied": False, "baseline_task_type": baseline,
                  "task_type": baseline, "used_dimensions": [], "minimum_confidence": min_confidence,
                  "policy_version": "domain_intent_v1"})
    reason = None
    domain = result.answers.get("domain")
    if result.error or not result.usable:
        reason = result.error or "no_prediction"
    elif baseline == "multimodal":
        reason = "image_constraint"
    elif not domain or domain.abstained or domain.value not in {"general", "coding", "debugging", "math", "science", "research"}:
        reason = "domain_unavailable"
    elif domain.confidence is None or domain.confidence < min_confidence:
        reason = "domain_low_confidence"
    elif mode == "shadow":
        reason = "shadow_only"
    elif mode != "enforce":
        reason = "decision_mode_disabled"
    if reason:
        trace["fallback_reason"] = reason
        return ClassificationContext(baseline, trace)
    task = domain.value
    used = ["domain"]
    intent = result.answers.get("intent_primary")
    if (task == "coding" and intent and not intent.abstained and intent.value in {"fix", "debug"}
            and intent.confidence is not None and intent.confidence >= min_confidence):
        task = "debugging"
        used.append("intent_primary")
    trace.update({"applied": True, "task_type": task, "used_dimensions": used, "fallback_reason": None})
    return ClassificationContext(task, trace)
