from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .learned_router import LearnedRouter
from .classification_input import _resolve_current_index, serialize_messages

from .models import ChatMessage


# Patterns the task classifier needs to recognise before it counts keywords. Hoisted out
# of the function so ``classify_text`` reads as a sequence of rules, and so they are
# compiled once rather than per request.
_N_PLUS_ONE_RE = re.compile(r"\bn\s*\+?\s*1\b|\bn\s*plus\s+one\b")
_ONE_QUERY_PER_RE = re.compile(r"\bone\s+(?:sql\s+)?quer(?:y|ies)\s+per\b")


TASK_KEYWORDS = {
    "debugging": [
        "bug",
        "debug",
        "failing test",
        "traceback",
        "stack trace",
        "vulnerability",
        "security",
        "exploit",
        "cve",
        "review",
        "operator mistake",
        "truncates decimals",
    ],
    "coding": [
        "code",
        "implement",
        "function",
        "class",
        "api",
        "repo",
        "patch",
        "test",
        "python",
        "typescript",
        "javascript",
        "sql",
    ],
    "math": [
        "compute",
        "prove",
        "derive",
        "solve",
        "equation",
        "fraction",
        "ratio",
        "arithmetic",
        "calculate",
        "integral",
        "probability",
        "geometry",
        "theorem",
        "aime",
        "olympiad",
    ],
    "science": [
        "biology",
        "chemistry",
        "physics",
        "gravity",
        "gravitational",
        "acceleration",
        "si unit",
        "molecule",
        "protein",
        "clinical",
        "experiment",
        "scientific",
    ],
    "research": [
        "research",
        "literature",
        "paper",
        "citation",
        "patent",
        "survey",
        "compare",
        "synthesize",
    ],
    "multimodal": [],
}


PREFERRED_WORKERS: Dict[str, List[str]] = {
    "debugging": ["claude-opus-4-5", "claude-sonnet-4-6", "gpt-5.5", "claude-opus-5-5", "claude-sonnet-5-5"],
    # Order re-measured on LCB-80 (2026-07-11): gpt-5.5 87.5, gpt-5.6-terra 83.8,
    # gemini-3.5-flash 76.2 (80.0 on 07-06), gpt-5.3-codex currently 402-dead on this
    # key (gateway pre-flight balance gate; historically ~80 — restore rank after
    # top-up if remeasured), sonnet-4-6 fallback. Exec-guided takes the top 3.
    "coding": ["gpt-5.5", "openai/gpt-5.6-terra", "gemini-3.5-flash", "gpt-5.3-codex", "claude-sonnet-4-6", "claude-sonnet-5-5", "claude-opus-5-5"],
    "math": ["gpt-5.5", "gemini-3.1-pro-preview", "claude-opus-4-5", "claude-opus-5-5", "claude-sonnet-5-5"],
    # Ordering set from the GPQA-main development split (disjoint from GPQA-Diamond):
    # gpt-5.5 was best-or-tied across every family, so text-knowledge science leads with it.
    "science": ["gpt-5.5", "gemini-3.1-pro-preview", "gemini-2.5-pro", "claude-opus-4-5", "claude-opus-5-5", "claude-sonnet-5-5"],
    "research": ["gpt-5.5", "gemini-3.1-pro-preview", "claude-opus-4-5", "claude-opus-5-5", "claude-sonnet-5-5"],
    # Measured on CharXiv-50 (2026-07-06, dual-judge verified): gpt-5.5 88-90 (beats
    # Fugu 86.6), claude-opus-4-8 76, claude-sonnet-4-6 75, claude-sonnet-5 70.
    "multimodal": ["gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6", "qwen3-vl-plus"],
    # Current Anthropic-only pools need task-specific fallbacks too. These appended
    # versions are deployment defaults, not a new benchmark-derived ranking.
    "general": ["gpt-5.4-mini", "gpt-5.4-nano", "gemini-3-flash-preview", "claude-haiku-5-5", "claude-sonnet-5-5"],
}

# Fallback when neither the task-preferred workers nor the fast-chat "general"
# workers are available in the pool (e.g. an all-frontier benchmark pool). Prefer
# strong reasoning models in dev-measured strength order instead of an arbitrary
# alphabetical pick.
STRONG_GENERALIST_WORKERS: List[str] = [
    "gpt-5.5",
    "gemini-3.1-pro-preview",
    "claude-opus-4-8",
    "claude-opus-4-5",
    "gemini-3-pro-preview",
    "gpt-5.4",
]


# Long-context lane (wired 2026-07-11). Measured: MRCR full-range stratified
# 2/4/8-needle 20K-680K tokens — gemini-3.1-pro-preview 87.5 (0 errors) vs
# gemini-2.5-pro 75.2; LongBench-v2 45 contexts — gemini-2.5-pro 75 >
# claude-sonnet-5 70 > gpt-5.5 64. Non-gemini workers cap ~128K tokens and degrade
# well before the cap, so past the threshold the ranked task lane is overridden.
# Coding/debugging are excluded: repo work stays in the exec-guided lane.
LONG_CONTEXT_WORKERS: List[str] = [
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
]
LONG_CONTEXT_CHARS = int(os.getenv("X1_LONG_CONTEXT_CHARS", "48000"))
# Past ~100K tokens the non-gemini workers exceed their context caps entirely, so the
# override applies regardless of task type (a coding prompt that big cannot run
# anywhere else either).
LONG_CONTEXT_HARD_CHARS = int(os.getenv("X1_LONG_CONTEXT_HARD_CHARS", "400000"))


def known_text_workers() -> List[str]:
    """Curated set of chat-callable text workers the router trusts.

    Used to give the Conductor a clean worker menu instead of the full provider
    inventory (which includes image/video/embedding models that 404 on chat).
    """
    seen: List[str] = []
    for models in PREFERRED_WORKERS.values():
        for model in models:
            if model not in seen:
                seen.append(model)
    for model in STRONG_GENERALIST_WORKERS:
        if model not in seen:
            seen.append(model)
    return seen


def ranked_workers(task_type: str, available_models: Iterable[str], top_k: int = 3) -> List[str]:
    """Up to top_k distinct available workers for a task, best first.

    Preferred workers for the task, then general, then strong generalists.
    Used to assemble a diverse pool for voting/ensemble strategies.
    """
    available = {m for m in available_models if m}
    ranked: List[str] = []
    for source in (PREFERRED_WORKERS.get(task_type, []), PREFERRED_WORKERS.get("general", []), STRONG_GENERALIST_WORKERS):
        for model in source:
            if model in available and model not in ranked:
                ranked.append(model)
                if len(ranked) >= top_k:
                    return ranked
    for model in sorted(available):
        if model not in ranked:
            ranked.append(model)
            if len(ranked) >= top_k:
                break
    return ranked


@dataclass(frozen=True)
class RouterDecision:
    task_type: str
    worker_model: str
    mode: str
    reason: str


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif _is_image_block(block):
                    parts.append("[image]")
        return "\n".join(parts)
    return "" if content is None else str(content)


def has_image_content(content: Any) -> bool:
    if isinstance(content, list):
        return any(isinstance(block, dict) and _is_image_block(block) for block in content)
    return False


def _is_image_block(block: Dict[str, Any]) -> bool:
    return (
        isinstance(block.get("image_url"), dict)
        or block.get("type") in {"image_url", "input_image"}
        or isinstance(block.get("image"), dict)
    )


def messages_to_text(messages: Iterable[ChatMessage]) -> str:
    return "\n".join(content_to_text(message.content) for message in messages)


def messages_to_classification_text(messages: Iterable[ChatMessage]) -> str:
    return serialize_messages(list(messages)).text


def is_multiple_choice(text: str) -> bool:
    """True for exam-style multiple-choice questions.

    These are hard knowledge tasks that must not be routed to fast-chat workers.
    Requires at least three lettered options (A/B/C[/D]) presented as choices, or
    an explicit boxed-letter answer instruction.
    """
    if re.search(r"\\boxed\{\s*\(?[a-d]\)?\s*\}", text):
        return True
    option_letters = set(re.findall(r"(?m)^\s*\(?([a-d])[\).]\s+\S", text))
    return len(option_letters) >= 3


def current_request_text(messages: Iterable[ChatMessage]) -> Optional[str]:
    """Serialize the current turn on its own, or ``None`` when there is no user turn.

    Uses the same serializer as the full read, so rendering rules cannot drift; only the
    history prefix is left out.
    """
    message_list = list(messages)
    if not message_list:
        return None
    try:
        index = _resolve_current_index(message_list, None)
    except Exception:
        return None
    if not (0 <= index < len(message_list)):
        return None
    message = message_list[index]
    try:
        return serialize_messages(
            [
                {
                    "role": str(getattr(message, "role", "user") or "user"),
                    "content": getattr(message, "content", ""),
                }
            ],
            current_user_index=0,
        ).text
    except Exception:
        return None


def classify_text(text: str) -> Optional[str]:
    """Keyword classification of an already-serialized, lower-cased text.

    Returns ``None`` when nothing matched, leaving the caller to decide whether to try a
    wider read or fall through to ``general``. Keeping the "no evidence" case distinct
    from ``general`` is what makes the two-pass read in :func:`classify_task` possible.
    """
    if "[image]" in text:
        return "multimodal"
    if is_multiple_choice(text):
        # A lettered multiple-choice exam question is a knowledge task; keep it out
        # of the fast-chat "general" bucket regardless of keyword content.
        return "science"
    if (
        _N_PLUS_ONE_RE.search(text)
        or _ONE_QUERY_PER_RE.search(text)
    ) and any(signal in text for signal in ["query", "queries", "database", "sql", "performance problem"]):
        return "debugging"
    if "compare" in text and any(signal in text for signal in ["rate limit", "rate limits", "quota", "quotas", "api design"]):
        return "research"
    scores: Dict[str, int] = {}
    for task, keywords in TASK_KEYWORDS.items():
        scores[task] = sum(1 for keyword in keywords if keyword in text)
    task, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return None
    return task


def classify_task(messages: Iterable[ChatMessage]) -> str:
    """Classify the request's task family from keyword evidence.

    The serialized text glues the visible history onto the current request, and the
    keyword tables are English substring lists. On a long non-English conversation the
    earlier turns can therefore outvote the current one: measured on a 200-request
    sample, dropping the history changes the verdict for 16.5% of requests, and for
    Chinese requests the change is almost always ``coding -> general`` (11 of 12),
    because English code words in the history were supplying the only evidence.

    So the current turn is scored first, and the full text (history included) is only
    consulted when the current turn matches nothing. That keeps requests whose intent
    genuinely depends on the conversation -- ``requires_prior_user_context`` is a real
    case in the label taxonomy -- while removing the history's ability to override a
    current turn that carries its own evidence.
    """
    message_list = list(messages)
    current_text = current_request_text(message_list)
    if current_text is not None:
        current_label = classify_text(current_text.lower())
        if current_label is not None:
            return current_label
    full_text = messages_to_classification_text(message_list).lower()
    fallback_label = classify_text(full_text)
    return fallback_label if fallback_label is not None else "general"


def choose_worker(task_type: str, available_models: Iterable[str], default_worker: Optional[str] = None) -> str:
    available_list = sorted({model for model in available_models if model})
    available = set(available_list)
    if default_worker and default_worker in available:
        return default_worker
    for candidate in PREFERRED_WORKERS.get(task_type, []):
        if candidate in available:
            return candidate
    for candidate in PREFERRED_WORKERS["general"]:
        if candidate in available:
            return candidate
    for candidate in STRONG_GENERALIST_WORKERS:
        if candidate in available:
            return candidate
    if available_list:
        return available_list[0]
    else:
        raise ValueError("No AnyInt worker models are available")


def select_worker_for_task(
    messages: Iterable[ChatMessage],
    available_models: Iterable[str],
    *,
    default_worker: Optional[str] = None,
    router_mode: str = "heuristic",
    learned_router: Optional[LearnedRouter] = None,
    coordinator: Any = None,
    classification_context: Any = None,
) -> RouterDecision:
    if router_mode not in {"heuristic", "learned_stub", "learned", "trinity"}:
        raise ValueError(f"Unsupported router mode: {router_mode}")
    message_list = list(messages)
    available_models = list(available_models)
    task_type = classification_context.task_type if classification_context is not None else classify_task(message_list)
    prompt_chars = sum(len(content_to_text(m.content)) for m in message_list)
    threshold = LONG_CONTEXT_HARD_CHARS if task_type in ("coding", "debugging") else LONG_CONTEXT_CHARS
    if prompt_chars >= threshold:
        available = {model for model in available_models if model}
        for candidate in LONG_CONTEXT_WORKERS:
            if candidate in available:
                return RouterDecision(
                    task_type=task_type,
                    worker_model=candidate,
                    mode=router_mode,
                    reason=f"long-context override: {prompt_chars} chars >= {threshold}",
                )
    if router_mode == "trinity" and coordinator is not None:
        worker = coordinator.predict_worker(message_list, available_models)
        if worker:
            return RouterDecision(
                task_type=task_type,
                worker_model=worker,
                mode=router_mode,
                reason="trinity coordinator (0.5B head) selection",
            )
    if router_mode == "learned" and learned_router is not None:
        # features are only used by the learned router — compute lazily so the default
        # heuristic/trinity paths skip a full-prompt feature scan (~0.45ms/req)
        features = extract_prompt_features(message_list)
        prediction = learned_router.predict(features, available_models)
        if prediction is not None:
            return RouterDecision(
                task_type=task_type,
                worker_model=prediction.worker_model,
                mode=router_mode,
                reason=f"learned router matched {prediction.matched_key} confidence={prediction.confidence:.2f}",
            )
    worker = choose_worker(task_type, available_models, default_worker)
    reason = "keyword heuristic"
    if classification_context is not None and classification_context.trace.get("applied"):
        return RouterDecision(task_type=task_type, worker_model=worker, mode="mmbert",
                              reason="mmbert domain/intent policy; Worker preference list")
    if router_mode == "learned_stub":
        reason = "learned_stub fallback to keyword heuristic"
    if router_mode == "learned":
        reason = "learned router fallback to keyword heuristic"
    if router_mode == "trinity":
        reason = "trinity coordinator fallback to keyword heuristic"
    return RouterDecision(task_type=task_type, worker_model=worker, mode=router_mode, reason=reason)


def extract_prompt_features(
    messages: Iterable[ChatMessage],
    *,
    requested_model: Optional[str] = None,
    tool_count: int = 0,
) -> Dict[str, Any]:
    message_list = list(messages)
    text = messages_to_classification_text(message_list)
    lower = text.lower()
    code_signals = ["def ", "class ", "```", "traceback", "pytest", "function", "typescript", "javascript", "python"]
    math_signals = ["prove", "derive", "equation", "integral", "probability", "*", "+", "="]
    security_signals = ["security", "vulnerability", "exploit", "cve", "injection", "auth"]
    research_signals = ["research", "paper", "citation", "compare", "survey", "literature"]
    return {
        "schema_version": 1,
        "requested_model": requested_model,
        "task_type": classify_task(message_list),
        "prompt_chars": len(text),
        "message_count": len(message_list),
        "has_code_signal": any(signal in lower for signal in code_signals),
        "has_math_signal": any(signal in lower for signal in math_signals),
        "has_security_signal": any(signal in lower for signal in security_signals),
        "has_research_signal": any(signal in lower for signal in research_signals),
        "image_block_count": text.count("[image]"),
        "tool_count": tool_count,
    }
