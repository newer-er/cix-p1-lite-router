from __future__ import annotations

import time
from dataclasses import dataclass, field
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .anyint_client import AnyIntClient
from .router import PREFERRED_WORKERS


@dataclass
class ModelObservation:
    calls: int = 0
    failures: int = 0
    total_latency_ms: int = 0
    total_cost: float = 0.0
    last_error_type: Optional[str] = None

    @property
    def avg_latency_ms(self) -> int:
        return int(self.total_latency_ms / self.calls) if self.calls else 0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.calls if self.calls else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0


@dataclass
class ModelCapability:
    id: str
    provider_family: str
    model_family: str
    modalities_in: List[str]
    modalities_out: List[str]
    task_strengths: Dict[str, float]
    supports_tools: Optional[bool] = None
    context_window: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    observation: ModelObservation = field(default_factory=ModelObservation)

    @property
    def family(self) -> str:
        return self.provider_family

    @property
    def modalities(self) -> List[str]:
        return sorted(set(self.modalities_in) | set(self.modalities_out))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provider_family": self.provider_family,
            "model_family": self.model_family,
            "modalities_in": list(self.modalities_in),
            "modalities_out": list(self.modalities_out),
            "modalities": list(self.modalities),
            "task_strengths": dict(sorted(self.task_strengths.items())),
            "supports_tools": self.supports_tools,
            "context_window": self.context_window,
            "raw": self.raw,
            "observed": {
                "calls": self.observation.calls,
                "failures": self.observation.failures,
                "avg_latency_ms": self.observation.avg_latency_ms,
                "avg_cost": self.observation.avg_cost,
                "failure_rate": self.observation.failure_rate,
                "last_error_type": self.observation.last_error_type,
            },
        }


class ModelRegistry:
    def __init__(self, client: AnyIntClient, ttl_seconds: float = 300.0, observation_db: Optional[str] = None) -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.observation_db = Path(observation_db) if observation_db else None
        self._models: List[Dict[str, Any]] = []
        self._capabilities: Dict[str, ModelCapability] = {}
        self._loaded_at = 0.0
        self._refreshing = False
        if self.observation_db is not None:
            self.observation_db.parent.mkdir(parents=True, exist_ok=True)
            self._init_observation_db()

    async def refresh(self) -> List[Dict[str, Any]]:
        models = await self.client.list_models()
        self._models = [model for model in models if model.get("id")]
        previous_capabilities = self._capabilities
        persisted = self._load_observation_rollups()
        self._capabilities = {}
        for model in self._models:
            model_id = str(model["id"])
            previous = previous_capabilities.get(model_id)
            capability = infer_capability(model_id, model, previous)
            if model_id in persisted:
                capability.observation = persisted[model_id]
            self._capabilities[model_id] = capability
        self._loaded_at = time.monotonic()
        return self._models

    async def models(self) -> List[Dict[str, Any]]:
        if not self._models:
            return await self.refresh()  # cold start: must block once
        # stale-while-revalidate: serve the cached list immediately and refresh in the
        # background, so no single request eats the periodic (5-min TTL) network refresh.
        if time.monotonic() - self._loaded_at > self.ttl_seconds and not self._refreshing:
            self._refreshing = True

            async def _bg_refresh() -> None:
                try:
                    await self.refresh()
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    self._refreshing = False

            try:
                import asyncio

                asyncio.create_task(_bg_refresh())
            except Exception:  # noqa: BLE001
                self._refreshing = False
        return self._models

    async def ids(self) -> List[str]:
        return [str(model["id"]) for model in await self.models()]

    async def capabilities(self) -> List[ModelCapability]:
        # _capabilities is built in model order during refresh(), so its values ARE the
        # per-model capabilities in order - no need to rebuild the id list and re-look-up
        # each one (this runs per request via ids_for()).
        await self.models()
        return list(self._capabilities.values())

    async def capability_map(self) -> Dict[str, ModelCapability]:
        await self.models()
        return dict(self._capabilities)

    async def ids_for(
        self,
        *,
        task_type: Optional[str] = None,
        requires_text_output: bool = True,
        required_input_modality: str = "text",
    ) -> List[str]:
        capabilities = await self.capabilities()
        filtered = [
            capability
            for capability in capabilities
            if (not requires_text_output or "text" in capability.modalities_out)
            and required_input_modality in capability.modalities_in
        ]
        return [
            capability.id
            for capability in sorted(
                filtered,
                key=lambda item: (
                    _health_penalty(item.observation),
                    -float(item.task_strengths.get(task_type or "general", item.task_strengths.get("general", 0.0))),
                    item.observation.failure_rate,
                    item.observation.avg_latency_ms,
                    item.id,
                ),
            )
        ]

    def record_observation(
        self,
        model: str,
        *,
        latency_ms: int = 0,
        cost: float = 0.0,
        failed: bool = False,
        error_type: Optional[str] = None,
        source: str = "api",
        request_model: Optional[str] = None,
        task_type: Optional[str] = None,
        role: Optional[str] = None,
        route_kind: Optional[str] = None,
    ) -> None:
        capability = self._capabilities.get(model)
        if capability is None:
            capability = infer_capability(model, {"id": model}, None)
            self._capabilities[model] = capability
        observation = capability.observation
        safe_latency = max(0, int(latency_ms))
        safe_cost = max(0.0, float(cost))
        observation.calls += 1
        observation.total_latency_ms += safe_latency
        observation.total_cost += safe_cost
        if failed:
            observation.failures += 1
            observation.last_error_type = error_type
        self._persist_observation(
            model=model,
            latency_ms=safe_latency,
            cost=safe_cost,
            failed=failed,
            error_type=error_type,
            source=source,
            request_model=request_model,
            task_type=task_type,
            role=role,
            route_kind=route_kind,
        )

    def export_capabilities(self) -> List[Dict[str, Any]]:
        return [capability.to_dict() for capability in sorted(self._capabilities.values(), key=lambda item: item.id)]

    def export_model_stats(self) -> List[Dict[str, Any]]:
        if self.observation_db is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT model_id, task_type, role, calls, failures, total_latency_ms,
                       total_cost, last_error_type, updated_at
                FROM model_stats
                ORDER BY model_id, task_type, role
                """
            ).fetchall()
        stats = []
        for row in rows:
            calls = int(row[3] or 0)
            failures = int(row[4] or 0)
            total_latency_ms = int(row[5] or 0)
            total_cost = float(row[6] or 0.0)
            stats.append(
                {
                    "model_id": row[0],
                    "task_type": row[1] or None,
                    "role": row[2] or None,
                    "calls": calls,
                    "failures": failures,
                    "avg_latency_ms": int(total_latency_ms / calls) if calls else 0,
                    "avg_cost": total_cost / calls if calls else 0.0,
                    "failure_rate": failures / calls if calls else 0.0,
                    "last_error_type": row[7],
                    "updated_at": row[8],
                }
            )
        return stats

    def _connect(self) -> sqlite3.Connection:
        if self.observation_db is None:
            raise RuntimeError("registry observation database is not configured")
        conn = sqlite3.connect(str(self.observation_db))
        # synchronous=NORMAL (per-connection) drops the fsync-per-commit that otherwise blocks
        # the event loop ~10x per observation write; WAL (set once at init) makes it safe.
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_observation_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    source TEXT NOT NULL,
                    request_model TEXT,
                    task_type TEXT,
                    role TEXT,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    error_type TEXT,
                    route_kind TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_stats (
                    model_id TEXT NOT NULL,
                    task_type TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT '',
                    calls INTEGER NOT NULL,
                    failures INTEGER NOT NULL,
                    total_latency_ms INTEGER NOT NULL,
                    total_cost REAL NOT NULL,
                    last_error_type TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (model_id, task_type, role)
                )
                """
            )

    def _persist_observation(
        self,
        *,
        model: str,
        latency_ms: int,
        cost: float,
        failed: bool,
        error_type: Optional[str],
        source: str,
        request_model: Optional[str],
        task_type: Optional[str],
        role: Optional[str],
        route_kind: Optional[str],
    ) -> None:
        if self.observation_db is None:
            return
        now = time.time()
        safe_task_type = task_type or ""
        safe_role = role or ""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_observations (
                    created_at, source, request_model, task_type, role, model_id,
                    status, latency_ms, cost, error_type, route_kind
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    source,
                    request_model,
                    task_type,
                    role,
                    model,
                    "error" if failed else "ok",
                    latency_ms,
                    cost,
                    error_type,
                    route_kind,
                ),
            )
            conn.execute(
                """
                INSERT INTO model_stats (
                    model_id, task_type, role, calls, failures, total_latency_ms,
                    total_cost, last_error_type, updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, task_type, role) DO UPDATE SET
                    calls = calls + 1,
                    failures = failures + excluded.failures,
                    total_latency_ms = total_latency_ms + excluded.total_latency_ms,
                    total_cost = total_cost + excluded.total_cost,
                    last_error_type = COALESCE(excluded.last_error_type, model_stats.last_error_type),
                    updated_at = excluded.updated_at
                """,
                (
                    model,
                    safe_task_type,
                    safe_role,
                    1 if failed else 0,
                    latency_ms,
                    cost,
                    error_type,
                    now,
                ),
            )

    def _load_observation_rollups(self) -> Dict[str, ModelObservation]:
        if self.observation_db is None or not self.observation_db.exists():
            return {}
        rollups: Dict[str, ModelObservation] = {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT model_id, SUM(calls), SUM(failures), SUM(total_latency_ms),
                       SUM(total_cost), MAX(updated_at)
                FROM model_stats
                GROUP BY model_id
                """
            ).fetchall()
            for model_id, calls, failures, latency, cost, _updated_at in rows:
                last_error_row = conn.execute(
                    """
                    SELECT last_error_type
                    FROM model_stats
                    WHERE model_id = ? AND last_error_type IS NOT NULL
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (model_id,),
                ).fetchone()
                rollups[str(model_id)] = ModelObservation(
                    calls=int(calls or 0),
                    failures=int(failures or 0),
                    total_latency_ms=int(latency or 0),
                    total_cost=float(cost or 0.0),
                    last_error_type=last_error_row[0] if last_error_row else None,
                )
        return rollups


def infer_capability(model_id: str, raw: Dict[str, Any], previous: Optional[ModelCapability] = None) -> ModelCapability:
    provider = provider_family(model_id)
    family = model_family(model_id)
    modalities_in, modalities_out = infer_modalities(model_id)
    strengths = infer_task_strengths(model_id)
    observation = previous.observation if previous is not None else ModelObservation()
    return ModelCapability(
        id=model_id,
        provider_family=provider,
        model_family=family,
        modalities_in=modalities_in,
        modalities_out=modalities_out,
        task_strengths=strengths,
        context_window=_context_window(raw),
        raw=dict(raw),
        observation=observation,
    )


def _health_penalty(observation: ModelObservation) -> int:
    if observation.calls >= 2 and observation.failure_rate >= 0.5:
        return 1
    return 0


def provider_family(model_id: str) -> str:
    lower = model_id.lower()
    if lower.startswith("gpt") or lower.startswith("text-embedding"):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("gemini"):
        return "google"
    if lower.startswith("grok"):
        return "xai"
    if lower.startswith("deepseek"):
        return "deepseek"
    if lower.startswith("qwen"):
        return "qwen"
    if lower.startswith("glm"):
        return "zhipu"
    if lower.startswith("kimi"):
        return "moonshot"
    if lower.startswith("ernie"):
        return "baidu"
    if lower.startswith("minimax"):
        return "minimax"
    if lower.startswith("veo"):
        return "google"
    if lower.startswith("wan"):
        return "wan"
    if lower.startswith("kling"):
        return "kling"
    if lower.startswith("hailuo"):
        return "hailuo"
    if lower.startswith("seedance"):
        return "seedance"
    if lower.startswith("happyhorse"):
        return "happyhorse"
    return lower.split("-", 1)[0].split("_", 1)[0]


def model_family(model_id: str) -> str:
    lower = model_id.lower()
    if lower.startswith("text-embedding") or "embedding" in lower:
        return "embedding"
    return lower.split("-", 1)[0].split("_", 1)[0]


def infer_modalities(model_id: str) -> tuple[List[str], List[str]]:
    lower = model_id.lower()
    if "embedding" in lower:
        return ["text"], ["embedding"]
    if any(marker in lower for marker in ["veo", "wan", "kling", "hailuo", "seedance", "t2v"]):
        return ["text"], ["video"]
    if "image" in lower and not any(marker in lower for marker in ["vl", "vision"]):
        return ["text", "image"], ["image"]
    modalities_in = ["text"]
    if any(marker in lower for marker in ["image", "vl", "vision"]):
        modalities_in.append("image")
    return modalities_in, ["text"]


def infer_task_strengths(model_id: str) -> Dict[str, float]:
    lower = model_id.lower()
    strengths: Dict[str, float] = {}
    for task_type, workers in PREFERRED_WORKERS.items():
        for index, worker in enumerate(workers):
            if worker == model_id:
                strengths[task_type] = max(strengths.get(task_type, 0.0), [1.0, 0.85, 0.75][min(index, 2)])
    if "codex" in lower or "code" in lower:
        strengths["coding"] = max(strengths.get("coding", 0.0), 1.0)
        strengths["debugging"] = max(strengths.get("debugging", 0.0), 0.85)
    if any(marker in lower for marker in ["opus", "sonnet", "gpt-5.5", "gemini-3.1-pro", "deepseek-v"]):
        strengths["reasoning"] = max(strengths.get("reasoning", 0.0), 0.85)
        strengths["research"] = max(strengths.get("research", 0.0), 0.75)
    if any(marker in lower for marker in ["flash", "nano", "haiku", "fast"]) or _has_variant(lower, "mini") or _has_variant(lower, "lite"):
        strengths["fast"] = max(strengths.get("fast", 0.0), 0.75)
        strengths["general"] = max(strengths.get("general", 0.0), 0.6)
    if any(marker in lower for marker in ["image", "vl", "vision"]):
        strengths["multimodal"] = max(strengths.get("multimodal", 0.0), 0.9)
    if "embedding" in lower:
        strengths["embedding"] = 1.0
    if not strengths:
        strengths["general"] = 0.4
    elif "general" not in strengths and "embedding" not in strengths:
        strengths["general"] = 0.4
    return dict(sorted(strengths.items()))


def _context_window(raw: Dict[str, Any]) -> Optional[int]:
    for key in ["context_window", "max_context_window", "context_length"]:
        value = raw.get(key)
        if isinstance(value, int):
            return value
    return None


def _has_variant(model_id: str, variant: str) -> bool:
    return (
        model_id == variant
        or model_id.startswith(f"{variant}-")
        or model_id.endswith(f"-{variant}")
        or f"-{variant}-" in model_id
        or f"_{variant}_" in model_id
    )
