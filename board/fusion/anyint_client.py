from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import httpx


class AnyIntError(RuntimeError):
    def __init__(self, message: str, *, model: Optional[str] = None, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.model = model
        self.status_code = status_code


class AnyIntClient:
    def __init__(self, api_key: str, base_url: str, timeout: float = 120.0, retries: int = 4) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)
        self._temperature_unsupported_models: set = set()
        # One pooled, keep-alive client reused across ALL requests instead of a fresh
        # AsyncClient (new TCP+TLS handshake) per call. Created lazily inside the loop.
        self._client: Optional[httpx.AsyncClient] = None
        self._limits = httpx.Limits(max_connections=128, max_keepalive_connections=64, keepalive_expiry=90.0)

    async def _get_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is not None:
            return client
        # Double-checked lazy init (GIL makes the assignment atomic); a rare concurrent
        # double-create is harmlessly reconciled by closing the loser.
        created = httpx.AsyncClient(timeout=self.timeout, limits=self._limits)
        if self._client is None:
            self._client = created
            return created
        await created.aclose()
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise AnyIntError("ANYINT_API_KEY is not configured")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def list_models(self) -> List[Dict[str, Any]]:
        response = await self._request("GET", f"{self.base_url}/models")
        if response.status_code >= 400:
            raise AnyIntError(f"AnyInt models request failed: status_code={response.status_code}", status_code=response.status_code)
        payload = response.json()
        return list(payload.get("data") or [])

    @staticmethod
    def _rejects_temperature(response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        try:
            text = response.text[:2000].lower()
        except Exception:
            return False
        if "temperature" not in text:
            return False
        return any(marker in text for marker in ("deprecated", "unsupported", "not support", "invalid"))

    # Transient statuses worth retrying. 402 is included: under concurrent load the
    # gateway returns rate-limit-style 402s that recover on retry (genuine balance
    # errors simply fail again after the retries). Retrying these closed a ~3-5%
    # benchmark gap where transient blips were scored as wrong answers.
    _RETRY_STATUS = frozenset({402, 408, 409, 425, 429})

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        import random

        last_response: Optional[httpx.Response] = None
        client = await self._get_client()
        for attempt in range(self.retries + 1):
            try:
                response = await client.request(method, url, headers=self._headers(), **kwargs)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt < self.retries:
                    await asyncio.sleep(min(8.0, 0.5 * (2**attempt)) * (0.5 + random.random()))
                    continue
                raise
            last_response = response
            if response.status_code not in self._RETRY_STATUS and response.status_code < 500:
                return response
            if attempt < self.retries:
                # jittered exponential backoff so concurrent retries don't thunder together
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)) * (0.5 + random.random()))
        return last_response

    async def stream_chat_content(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """True token streaming: yield content deltas as the worker produces them.

        Sets stream:true and parses the SSE deltas, so time-to-first-token is the
        worker's first token, not the full response. Single-worker path only (fusion
        strategies that select/verify must buffer). Raises before the first yield on
        setup failure so the caller can fall back to the buffered path."""
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if max_tokens is not None:
            body["max_tokens"] = max(16, int(max_tokens))
        if temperature is not None and model not in self._temperature_unsupported_models:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        client = await self._get_client()
        async with client.stream("POST", f"{self.base_url}/chat/completions",
                                 headers=self._headers(), json=body) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise AnyIntError(f"AnyInt stream failed: status_code={resp.status_code}",
                                  model=model, status_code=resp.status_code)
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                    piece = delta.get("content")
                except (KeyError, IndexError, ValueError, TypeError):
                    continue
                if piece:
                    yield piece

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max(16, int(max_tokens))
        if temperature is not None and model not in self._temperature_unsupported_models:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None and _chat_compatible_tools(tools):
            body["tools"] = tools
        if tool_choice is not None and "tools" in body:
            body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None and "tools" in body:
            body["parallel_tool_calls"] = bool(parallel_tool_calls)

        started = time.monotonic()
        response = await self._request("POST", f"{self.base_url}/chat/completions", json=body)
        if "temperature" in body and self._rejects_temperature(response):
            self._temperature_unsupported_models.add(model)
            body.pop("temperature", None)
            response = await self._request("POST", f"{self.base_url}/chat/completions", json=body)
        latency_ms = int((time.monotonic() - started) * 1000)
        if response.status_code >= 400:
            raise AnyIntError(f"AnyInt chat request failed: status_code={response.status_code}", model=model, status_code=response.status_code)
        payload = response.json()
        payload["_x1_latency_ms"] = latency_ms
        return payload

    async def responses_create(
        self,
        model: str,
        input_value: Any,
        *,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "input": input_value,
            "stream": False,
        }
        if instructions is not None:
            body["instructions"] = instructions
        if max_output_tokens is not None:
            body["max_output_tokens"] = max(16, int(max_output_tokens))
        if temperature is not None and model not in self._temperature_unsupported_models:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None and tools is not None:
            body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None and tools is not None:
            body["parallel_tool_calls"] = bool(parallel_tool_calls)

        started = time.monotonic()
        response = await self._request("POST", f"{self.base_url}/responses", json=body)
        if "temperature" in body and self._rejects_temperature(response):
            self._temperature_unsupported_models.add(model)
            body.pop("temperature", None)
            response = await self._request("POST", f"{self.base_url}/responses", json=body)
        latency_ms = int((time.monotonic() - started) * 1000)
        if response.status_code >= 400:
            raise AnyIntError(f"AnyInt responses request failed: status_code={response.status_code}", model=model, status_code=response.status_code)
        payload = response.json()
        payload["_x1_latency_ms"] = latency_ms
        return payload


def build_anyint_client(api_key: str, base_url: str, timeout: float = 120.0, retries: int = 4) -> AnyIntClient:
    from urllib.parse import urlsplit

    path = urlsplit(base_url).path.rstrip("/")
    if path.endswith("/anthropic") or path.endswith("/anthropic/v1"):
        from .anthropic_client import AnthropicClient

        return AnthropicClient(api_key, base_url, timeout, retries)
    return AnyIntClient(api_key, base_url, timeout, retries)


def extract_chat_text(payload: Dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def extract_chat_tool_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        tool_calls = payload["choices"][0]["message"].get("tool_calls")
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(tool_calls, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = function.get("name")
        arguments = function.get("arguments", "")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, str):
            arguments = json_dumps(arguments)
        normalized.append(
            {
                "id": str(tool_call.get("id") or f"call_{index}"),
                "type": str(tool_call.get("type") or "function"),
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return normalized


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _chat_compatible_tools(tools: List[Dict[str, Any]]) -> bool:
    return all(isinstance(tool, dict) and tool.get("type") == "function" for tool in tools)


def extract_response_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    parts: List[str] = []
    output = payload.get("output") or []
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)
