"""Serve the demo's single-turn auto route using the board classifier."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import os
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from fusion.anyint_client import AnyIntError, build_anyint_client
from fusion.classification_policy import classification_context
from fusion.classifier_client import MmbertBackend
from fusion.models import ChatCompletionRequest
from fusion.registry import ModelRegistry
from fusion.router import classify_task, select_worker_for_task


SYSTEM_PROMPT = (
    "You are x1-lite, a balanced fusion router. Answer directly and accurately. "
    "Prefer concise, actionable output. If the task is code or debugging, be precise."
)


def create_app(classifier_url: str, upstream_url: str, api_key: str) -> FastAPI:
    backend = MmbertBackend(classifier_url, timeout=40, health_timeout=2)
    upstream = build_anyint_client(api_key, upstream_url, timeout=45, retries=0)
    registry = ModelRegistry(upstream)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            backend.close()
            await upstream.aclose()

    app = FastAPI(title="AnyInt Lite Router Board Demo", lifespan=lifespan)

    @app.get("/health")
    async def health():
        ready = await asyncio.to_thread(backend.health)
        return {
            "status": "ok" if ready else "degraded",
            "classification_ready": ready,
            "upstream_configured": bool(api_key),
        }

    @app.post("/openai/v1/chat/completions")
    async def chat(request: ChatCompletionRequest):
        if request.model != "auto" or request.stream or request.tools or len(request.messages) != 1:
            raise HTTPException(400, detail="demo_requires_one_text_request_with_auto")
        message = request.messages[0]
        if message.role != "user" or not isinstance(message.content, str) or not 0 < len(message.content.strip()) <= 12_000:
            raise HTTPException(400, detail="demo_invalid_user_message")
        if not api_key:
            raise HTTPException(503, detail="upstream_not_configured")

        prediction = await asyncio.to_thread(backend.classify, request.messages)
        context = classification_context(
            classify_task(request.messages), prediction, mode="enforce", min_confidence=0.7,
        )
        try:
            available = await registry.ids_for(task_type=context.task_type)
            decision = select_worker_for_task(request.messages, available, classification_context=context)
            output = await upstream.chat_completion(
                decision.worker_model,
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message.content}],
                max_tokens=min(640, max(16, request.max_tokens or 640)),
            )
        except httpx.TimeoutException as error:
            raise HTTPException(504, detail="upstream_timeout") from error
        except AnyIntError as error:
            raise HTTPException(502, detail={"error": "upstream_rejected", "status": error.status_code}) from error
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(502, detail="upstream_unavailable") from error
        if not isinstance(output, dict) or not output.get("choices"):
            raise HTTPException(502, detail="invalid_upstream_response")

        elapsed = output.pop("_x1_latency_ms", None)
        output["x1_route"] = {
            "task_type": decision.task_type,
            "worker_model": decision.worker_model,
            "calls": [{"model": decision.worker_model, "latency_ms": elapsed}],
            "budget": {"worker_calls": 1},
            "workflow": {"classification": context.trace},
        }
        return output

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-url", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    url = os.environ.get("ANYINT_BASE_URL", "https://api.anyint.ai/openai/v1")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        parser.error("ANYINT_BASE_URL 必须是 HTTP(S) API 根地址")
    api_key = os.environ.get("ANYINT_API_KEY", "").strip()
    if not api_key:
        parser.error("设置 ANYINT_API_KEY 后再启动板端 Router")
    uvicorn.run(create_app(args.classifier_url, url, api_key), host="127.0.0.1", port=args.port, access_log=False)
