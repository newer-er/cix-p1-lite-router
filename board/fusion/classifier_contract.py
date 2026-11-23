"""Fixed labels and validated wire format for the trained Lite classifier."""
from __future__ import annotations

import math
from typing import Any

from .classification_input import INPUT_SCHEMA_VERSION

CLASSIFIER_SCHEMA = "lite_classifier.v1"
TAXONOMY_VERSION = "labeling_taxonomy_v2"
LABELS = {
    "domain": ["general", "coding", "debugging", "math", "science", "research"],
    "intent_primary": ["generate", "fix", "debug", "review", "explain", "compare", "summarize", "search", "plan", "transform", "other"],
    "complexity": ["easy", "medium", "hard"],
    "context_dependency": ["self_contained", "requires_prior_user_context", "depends_on_model_output", "requires_external_context"],
    "output_format": ["freeform", "code", "structured_data", "table", "list", "other"],
    **{d: ["yes", "no", "uncertain"] for d in ("requires_tools", "requires_code_execution", "requires_retrieval", "requires_long_context", "requires_vision")},
}
MULTI_LABELS = {"intent_secondary": LABELS["intent_primary"].copy()}


def validate_result(body: Any) -> dict:
    """Reject malformed probabilities instead of silently routing on defaults."""
    if not isinstance(body, dict) or body.get("schema_version") != CLASSIFIER_SCHEMA:
        raise ValueError("classifier_response_schema")
    if body.get("input_schema_version") != INPUT_SCHEMA_VERSION or body.get("taxonomy_version") != TAXONOMY_VERSION:
        raise ValueError("classifier_response_version")
    if not isinstance(body.get("model_version"), str) or not body["model_version"]:
        raise ValueError("classifier_model_version_missing")
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("classifier_answers_missing")
    for dim, labels in {**LABELS, **MULTI_LABELS}.items():
        item = answers.get(dim, {})
        probs = item.get("probabilities", {})
        if set(probs) != set(labels):
            raise ValueError("classifier_probability_labels")
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
            raise ValueError("classifier_probabilities_invalid")
        if dim in LABELS:
            best = max(labels, key=lambda label: probs[label])
            if abs(sum(probs.values()) - 1) > 1e-4 or item.get("value") != best:
                raise ValueError("classifier_probability_distribution")
            confidence = item.get("confidence")
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or abs(confidence - probs[best]) > 1e-5:
                raise ValueError("classifier_confidence_invalid")
        elif item.get("value") != [label for label in labels if probs[label] >= 0.5]:
            raise ValueError("classifier_multilabel_invalid")
    return body
