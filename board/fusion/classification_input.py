from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


INPUT_SCHEMA_VERSION = "classification_input.v1"
DEFAULT_MAX_CHARS = 4096
CURRENT_SHARE = 0.75
TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
IMAGE_BLOCK_TYPES = {"image", "image_url", "input_image"}
REASONING_BLOCK_TYPES = {"thinking", "reasoning", "redacted_thinking"}
OMITTED_ROLES = {"system", "developer"}

# --- token estimation ---------------------------------------------------------
#
# The character budget above is what the serializer enforces; token counts are what
# every model constraint is written in (Laya 1024/8192, Kev 384/1024, Jev 32000).
# A single ``chars / 4`` cannot bridge the two: measured on this corpus with the
# mmBERT tokenizer (``m2-decision-envs/calibrate_tokens.py``), it underestimates
# every language -- Chinese by ~50%, English by ~29%.
#
# These ratios are measured characters-per-token, used as ``ceil(chars / ratio)``.
# They come from a real tokenizer run offline; this module stays dependency-free.
# Re-measure and bump ``TOKEN_ESTIMATE_METHOD`` when the corpus or the target
# tokenizer changes.
TOKEN_ESTIMATE_METHOD = "mmbert_chars_per_token_v1"

# language -> characters per token, measured p50 over 5,230 records.
#
# Grouped by ``state_language``, NOT by the current turn's language: the ratio is
# applied to the whole serialized state text, so it must be measured on the same
# population it is applied to. (8.4% of records get a different label under the two
# rules; measuring per-turn and applying per-state would be the exact class of
# mismatch this calibration exists to remove.)
CHARS_PER_TOKEN = {
    "zh": 2.01,
    "mixed": 2.97,
    "en": 2.93,
}
CHARS_PER_TOKEN_DEFAULT = 2.97

# The pre-calibration estimate, kept so older reports stay readable.
LEGACY_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class ClassificationInput:
    text: str
    current_text: str
    history: List[Dict[str, str]]
    metadata: Dict[str, Any]
    truncation: Dict[str, Any]
    schema_version: str = INPUT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _tool_name(tool: Any) -> Optional[str]:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    if tool.get("name"):
        return str(tool["name"])
    return None


def _render_content(content: Any, counts: Counter[str]) -> str:
    if isinstance(content, str):
        counts["text_blocks"] += 1
        return content
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            counts["text_blocks"] += 1
            parts.append(block)
            continue
        if not isinstance(block, dict):
            counts["unknown_blocks"] += 1
            continue
        block_type = str(block.get("type") or "unknown")
        counts[f"{block_type}_blocks"] += 1
        if block_type in TEXT_BLOCK_TYPES:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type in IMAGE_BLOCK_TYPES:
            parts.append("[image]")
        elif block_type == "tool_result":
            nested = _render_content(block.get("content"), counts).strip()
            parts.append("[tool_result]" + (f"\n{nested}" if nested else ""))
        elif block_type == "tool_use":
            parts.append(f"[tool_use:{block.get('name') or '?'}]")
        elif block_type in REASONING_BLOCK_TYPES:
            counts["thinking_blocks_omitted"] += 1
        else:
            counts["unknown_blocks_omitted"] += 1
    return "\n".join(part for part in parts if part)


def _has_human_content(message: Any) -> bool:
    if str(_field(message, "role", "")) != "user":
        return False
    content = _field(message, "content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, str) and block.strip():
            return True
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type in IMAGE_BLOCK_TYPES:
            return True
        if block_type in TEXT_BLOCK_TYPES and str(block.get("text") or "").strip():
            return True
    return False


def _resolve_current_index(messages: Sequence[Any], explicit: Optional[int]) -> int:
    if isinstance(explicit, int) and 0 <= explicit < len(messages):
        return explicit
    for index in range(len(messages) - 1, -1, -1):
        if _has_human_content(messages[index]):
            return index
    for index in range(len(messages) - 1, -1, -1):
        if str(_field(messages[index], "role", "")) == "user":
            return index
    return len(messages) - 1 if messages else -1


def _head_tail(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    if limit <= 0:
        return "", bool(text)
    marker = "\n[...truncated...]\n"
    if limit <= len(marker):
        return text[:limit], True
    usable = max(0, limit - len(marker))
    head = int(usable * 0.65)
    tail = usable - head
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def _history_within_budget(
    history: Sequence[Dict[str, str]], budget: int
) -> Tuple[List[Dict[str, str]], bool]:
    if budget <= 0 or not history:
        return [], bool(history)
    kept_reversed: List[Dict[str, str]] = []
    used = 0
    truncated = False
    for entry in reversed(history):
        prefix = f"[history role={entry['role']}]\n"
        remaining = budget - used - len(prefix)
        if remaining <= 0:
            truncated = True
            break
        text, item_truncated = _head_tail(entry["text"], remaining)
        kept_reversed.append({"role": entry["role"], "text": text})
        used += len(prefix) + len(text) + 1
        truncated = truncated or item_truncated
        if item_truncated:
            break
    kept = list(reversed(kept_reversed))
    return kept, truncated or len(kept) < len(history)


def _language(text: str) -> str:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_words = len(re.findall(r"\b[A-Za-z]{2,}\b", text))
    if cjk and latin_words >= max(8, cjk // 2):
        return "mixed"
    if cjk:
        return "zh"
    if latin_words:
        return "en"
    return "other"


def _bucket_from_tokens(tokens: int) -> str:
    """Bucket label for a token count. Thresholds are stable across method versions."""
    if tokens <= 1024:
        return "0_1k"
    if tokens <= 4096:
        return "1k_4k"
    if tokens <= 16384:
        return "4k_16k"
    if tokens <= 65536:
        return "16k_64k"
    return "64k_plus"


def estimate_tokens(chars: int, language: str, *, chars_per_token: float = None) -> int:
    """Estimate a token count from a character count, aware of the text's language.

    ``chars / 4`` is the English-centric default and is badly wrong elsewhere; the
    measured ratios in ``CHARS_PER_TOKEN`` replace it. When no calibrated ratio is
    passed, the language's table entry is used, falling back to the mixed ratio.
    """
    if chars <= 0:
        return 0
    ratio = chars_per_token
    if ratio is None:
        ratio = CHARS_PER_TOKEN.get(language, CHARS_PER_TOKEN_DEFAULT)
    if ratio <= 0:
        ratio = CHARS_PER_TOKEN_DEFAULT
    return math.ceil(chars / ratio)


def _context_bucket(chars: int, language: str = None) -> str:
    """Bucket a length, interpreting it in tokens.

    Kept character-taking for callers that pass a raw length, but the conversion is
    now language-aware. ``language=None`` uses the mixed-ratio default.
    """
    return _bucket_from_tokens(estimate_tokens(chars, language or "", chars_per_token=None))


def _legacy_context_bucket(chars: int) -> str:
    """The ``chars / 4`` bucket, for reports written before calibration."""
    return _bucket_from_tokens(math.ceil(chars / LEGACY_CHARS_PER_TOKEN))


def serialize_messages(
    messages: Iterable[Any],
    *,
    current_user_index: Optional[int] = None,
    tools: Optional[Iterable[Any]] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    privacy: Optional[Dict[str, Any]] = None,
) -> ClassificationInput:
    message_list = list(messages)
    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    current_index = _resolve_current_index(message_list, current_user_index)
    counts: Counter[str] = Counter()
    history_full: List[Dict[str, str]] = []
    current_full = ""
    system_chars = 0

    for index, message in enumerate(message_list):
        role = str(_field(message, "role", "unknown") or "unknown")
        if index > current_index:
            counts["post_current_messages_omitted"] += 1
            continue
        local_counts: Counter[str] = Counter()
        rendered = _render_content(_field(message, "content", ""), local_counts).strip()
        counts.update(local_counts)
        if role in OMITTED_ROLES:
            counts[f"{role}_messages_omitted"] += 1
            system_chars += len(rendered)
            continue
        if index == current_index:
            current_full = rendered
            continue
        if rendered and role in {"user", "assistant", "tool"}:
            history_full.append({"role": role, "text": rendered})

    current_header = "[current role=user]\n"
    history_exists = bool(history_full)
    current_budget = max_chars - len(current_header)
    if history_exists:
        current_budget = max(128, int(max_chars * CURRENT_SHARE) - len(current_header))
    current_text, current_truncated = _head_tail(current_full, current_budget)
    history_budget = max_chars - len(current_header) - len(current_text) - 1
    history, history_truncated = _history_within_budget(history_full, history_budget)
    history_text = "\n".join(
        f"[history role={entry['role']}]\n{entry['text']}" for entry in history
    )
    text = "\n".join(part for part in (history_text, current_header + current_text) if part)
    if len(text) > max_chars:
        overflow = len(text) - max_chars
        history_text = history_text[:-overflow] if overflow < len(history_text) else ""
        history_truncated = True
        text = "\n".join(part for part in (history_text, current_header + current_text) if part)

    tool_names = [name for name in (_tool_name(tool) for tool in (tools or [])) if name]
    modality = ["text"]
    image_count = sum(count for key, count in counts.items() if key in {"image_blocks", "image_url_blocks", "input_image_blocks"})
    if image_count:
        modality.append("image")
    full_chars = len(current_full) + sum(len(entry["text"]) for entry in history_full)
    findings = (privacy or {}).get("finding_counts") or {}
    # The token estimate covers the whole state (history + current), so judge the
    # language of that whole state rather than of the current turn alone -- the
    # ratios differ enough (zh 2.01 vs mixed 2.97) to matter, and the two labels
    # disagree on 8.4% of the corpus.
    state_language = _language(
        "\n".join([entry["text"] for entry in history_full] + [current_full])
    )
    metadata = {
        "current_user_index": current_index,
        "message_count": max(0, current_index + 1),
        "history_entry_count": len(history_full),
        "system_or_developer_chars_omitted": system_chars,
        "modality": modality,
        "image_block_count": image_count,
        "tool_result_block_count": counts.get("tool_result_blocks", 0),
        "tool_use_block_count": counts.get("tool_use_blocks", 0),
        "tool_names": tool_names,
        "tool_count": len(tool_names),
        "has_code_block": "```" in current_full,
        "has_structured_output_constraint": bool(re.search(r"\b(json|xml|yaml|csv)\b|表格|结构化", current_full, re.IGNORECASE)),
        "language": _language(current_full),
        "state_language": state_language,
        "context_bucket": _context_bucket(full_chars, state_language),
        "context_bucket_chars_v1": _legacy_context_bucket(full_chars),
        "token_estimate_method": TOKEN_ESTIMATE_METHOD,
        "estimated_full_tokens": estimate_tokens(full_chars, state_language),
        "estimated_serialized_tokens": estimate_tokens(len(text), state_language),
        "privacy_findings": dict(sorted(findings.items())),
        "risk_gates": {
            "contains_pii": any(key in findings for key in ("email", "phone", "id_card")),
            "contains_secret": any("token" in key or "secret" in key or "credential" in key or "key" in key for key in findings),
        },
    }
    truncation = {
        "max_chars": max_chars,
        "full_chars": full_chars,
        "serialized_chars": len(text),
        "current_original_chars": len(current_full),
        "current_serialized_chars": len(current_text),
        "history_original_entries": len(history_full),
        "history_serialized_entries": len(history),
        "current_truncated": current_truncated,
        "history_truncated": history_truncated,
        "post_current_messages_omitted": counts.get("post_current_messages_omitted", 0),
    }
    return ClassificationInput(
        text=text,
        current_text=current_text,
        history=history,
        metadata=metadata,
        truncation=truncation,
    )


def serialize_record(record: Dict[str, Any], *, max_chars: int = DEFAULT_MAX_CHARS) -> ClassificationInput:
    request = record.get("request") or {}
    conversation = request.get("conversation") or []
    messages = [
        {"role": entry.get("role", "unknown"), "content": entry.get("text", "")}
        for entry in conversation
        if isinstance(entry, dict)
    ]
    result = serialize_messages(
        messages,
        current_user_index=request.get("current_user_index"),
        max_chars=max_chars,
        privacy=record.get("privacy") if isinstance(record.get("privacy"), dict) else None,
    )
    modality = record.get("modality") or {}
    current_modality = modality.get("current_request") or {}
    image_count = int(current_modality.get("image_blocks", 0) or 0)
    tool_result_count = int(current_modality.get("tool_result_blocks", 0) or 0)
    tool_use_count = int(current_modality.get("tool_use_blocks", 0) or 0)
    if image_count and "image" not in result.metadata["modality"]:
        result.metadata["modality"].append("image")
    result.metadata["image_block_count"] = max(result.metadata["image_block_count"], image_count)
    result.metadata["tool_result_block_count"] = max(result.metadata["tool_result_block_count"], tool_result_count)
    result.metadata["tool_use_block_count"] = max(result.metadata.get("tool_use_block_count", 0), tool_use_count)
    attribution = (record.get("processing") or {}).get("attribution") or {}
    result.truncation["post_current_messages_omitted"] = max(
        result.truncation["post_current_messages_omitted"],
        int(attribution.get("post_current_message_count", 0) or 0),
    )
    result.metadata["source_history_entries_omitted"] = int(
        (modality.get("history") or {}).get("history_entries_omitted", 0) or 0
    )
    result.metadata["source_history_entries_truncated"] = int(
        (modality.get("history") or {}).get("history_entry_truncated", 0) or 0
    )
    return result


def audit_directory(input_dir: Path, *, limit: Optional[int] = None) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for path in sorted(input_dir.glob("*.json")):
        if limit is not None and counts["records"] >= limit:
            break
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            result = serialize_record(record)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors[type(exc).__name__] += 1
            continue
        counts["records"] += 1
        counts[f"language:{result.metadata['language']}"] += 1
        counts[f"state_language:{result.metadata['state_language']}"] += 1
        counts[f"context_bucket:{result.metadata['context_bucket']}"] += 1
        counts[f"context_bucket_chars_v1:{result.metadata['context_bucket_chars_v1']}"] += 1
        counts["current_truncated"] += int(result.truncation["current_truncated"])
        counts["history_truncated"] += int(result.truncation["history_truncated"])
        counts["has_image"] += int("image" in result.metadata["modality"])
        counts["has_tool_result"] += int(result.metadata["tool_result_block_count"] > 0)
        counts["post_current_omitted"] += int(result.truncation["post_current_messages_omitted"] > 0)
        for gate in ("contains_pii", "contains_secret"):
            counts[f"risk:{gate}"] += int(bool(result.metadata["risk_gates"].get(gate)))
    return {
        "schema_version": "classification_input_audit.v1",
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "limit": limit,
        # Which estimate dialect the bucket counts above were produced under. Reports
        # are compared across stages, so an unlabelled bucket distribution is not
        # interpretable on its own.
        "token_estimate_method": TOKEN_ESTIMATE_METHOD,
        "counts": dict(sorted(counts.items())),
        "errors": dict(sorted(errors.items())),
        "contains_request_text": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a content-free classification input audit")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = audit_directory(args.input_dir, limit=args.limit)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
