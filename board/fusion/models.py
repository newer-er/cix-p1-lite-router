from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


AUTO_MODEL = "auto"
VIRTUAL_MODELS = ("x1-lite", "x1-ultra", AUTO_MODEL)

_MODEL_DESCRIPTIONS = {
    "x1-lite": "Balanced Fugu-compatible fusion model backed by AnyInt workers.",
    "x1-ultra": "Quality-first multi-agent fusion model backed by AnyInt workers.",
    AUTO_MODEL: "Automatically routed X1 mode backed by the Lite strategy pipeline.",
}

_MODEL_DISPLAY_NAMES = {
    "x1-lite": "X1 Lite",
    "x1-ultra": "X1 Ultra",
    AUTO_MODEL: "X1 Auto",
}


def resolve_virtual_model(model_id: str) -> str:
    """Resolve the user-facing model alias to its current serving pipeline."""
    return "x1-lite" if model_id == AUTO_MODEL else model_id


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class ResponsesRequest(BaseModel):
    model: str
    input: Any
    instructions: Optional[str] = None
    stream: bool = False
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class WorkerCall(BaseModel):
    role: str
    model: str
    task: str
    content: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    latency_ms: int


class FusionResult(BaseModel):
    output_text: str
    worker_model: str
    task_type: str
    calls: List[WorkerCall]
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Dict[str, Any] = Field(default_factory=dict)
    route: Dict[str, Any] = Field(default_factory=dict)


def virtual_model_records() -> List[Dict[str, Any]]:
    now = int(time.time())
    return [_openai_model_record(model_id, now) for model_id in VIRTUAL_MODELS]


def codex_model_records() -> List[Dict[str, Any]]:
    return [_codex_model_record(model_id) for model_id in VIRTUAL_MODELS]


def _openai_model_record(model_id: str, created: int) -> Dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": "fusion",
        "display_name": _MODEL_DISPLAY_NAMES[model_id],
        "description": _MODEL_DESCRIPTIONS[model_id],
    }


def _codex_model_record(model_id: str) -> Dict[str, Any]:
    return {
        "slug": model_id,
        "display_name": _MODEL_DISPLAY_NAMES[model_id],
        "description": _MODEL_DESCRIPTIONS[model_id],
        "default_reasoning_level": None,
        "supported_reasoning_levels": [],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "availability_nux": {"message": ""},
        "upgrade": None,
        "base_instructions": "",
        "model_messages": {"instructions_template": "", "instructions_variables": {}},
        "supports_reasoning_summaries": False,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": False,
        "supports_image_detail_original": False,
        "context_window": 1000000,
        "max_context_window": 1000000,
        "comp_hash": "x1",
        "effective_context_window_percent": 100,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": False,
        "use_responses_lite": False,
    }
