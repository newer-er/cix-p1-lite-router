"""Anthropic Messages transport behind the router's existing worker interface.

Protocol reference: https://platform.claude.com/docs/en/api/messages/create
No upstream prompt, credential, or provider error body is included in errors.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .anyint_client import AnyIntClient, AnyIntError


def _blocks(content: Any) -> list[dict]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise AnyIntError("Unsupported Anthropic message content")
    result = []
    for block in content:
        if not isinstance(block, dict):
            raise AnyIntError("Unsupported Anthropic content block")
        kind = block.get("type")
        if kind in {"text", "input_text", "output_text"}:
            result.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "image_url":
            image = block.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str):
                raise AnyIntError("Invalid Anthropic image source")
            if url.startswith("data:"):
                try:
                    media, data = url[5:].split(";base64,", 1)
                except ValueError:
                    raise AnyIntError("Invalid Anthropic image source") from None
                source = {"type": "base64", "media_type": media, "data": data}
            else:
                source = {"type": "url", "url": url}
            result.append({"type": "image", "source": source})
        else:
            raise AnyIntError("Unsupported Anthropic content block")
    return result


def _messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    system, result = [], []
    for message in messages:
        role = message.get("role")
        content = _blocks(message.get("content"))
        if role in {"system", "developer"}:
            if any(block["type"] != "text" for block in content):
                raise AnyIntError("Anthropic system content must be text")
            system.extend(content)
            continue
        if role == "tool":
            if not message.get("tool_call_id"):
                raise AnyIntError("Anthropic tool result requires a call id")
            content = [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": content}]
            role = "user"
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = function.get("arguments", "{}")
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    if not isinstance(arguments, dict) or not call.get("id") or not function.get("name"):
                        raise ValueError
                except (ValueError, TypeError):
                    raise AnyIntError("Invalid Anthropic tool call") from None
                content.append({"type": "tool_use", "id": call["id"], "name": function["name"], "input": arguments})
        elif role != "user":
            raise AnyIntError("Unsupported Anthropic message role")
        if not content:
            continue
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(content)
        else:
            result.append({"role": role, "content": content})
    return system, result


class AnthropicClient(AnyIntClient):
    def __init__(self, api_key: str, base_url: str, timeout: float = 120.0, retries: int = 4):
        base_url = base_url.rstrip("/")
        super().__init__(api_key, base_url if base_url.endswith("/v1") else base_url + "/v1", timeout, retries)

    def _headers(self):
        if not self.api_key:
            raise AnyIntError("ANYINT_API_KEY is not configured")
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}

    async def list_models(self):
        models, cursor = [], None
        for _ in range(20):
            params = {"limit": 1000}
            if cursor:
                params["after_id"] = cursor
            response = await self._request("GET", self.base_url + "/models", params=params)
            if response.status_code >= 400:
                raise AnyIntError(f"Anthropic models request failed: status_code={response.status_code}", status_code=response.status_code)
            payload = response.json()
            models.extend(payload.get("data") or [])
            if not payload.get("has_more"):
                return models
            next_cursor = payload.get("last_id")
            if not next_cursor or next_cursor == cursor:
                raise AnyIntError("Invalid Anthropic model pagination")
            cursor = next_cursor
        raise AnyIntError("Anthropic model pagination limit exceeded")

    def _body(self, model, messages, *, max_tokens=None, temperature=None, top_p=None,
              tools=None, tool_choice=None, parallel_tool_calls=None):
        system, converted = _messages(messages)
        body = {"model": model, "messages": converted, "max_tokens": max(16, int(max_tokens)) if max_tokens is not None else 1024}
        if system:
            body["system"] = system
        if temperature is not None and model not in self._temperature_unsupported_models:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if tools:
            converted_tools = []
            for tool in tools:
                function = tool.get("function") or {}
                if tool.get("type") != "function" or not function.get("name"):
                    raise AnyIntError("Anthropic transport supports function tools only")
                item = {"name": function["name"], "input_schema": function.get("parameters") or {"type": "object", "properties": {}}}
                if function.get("description"):
                    item["description"] = function["description"]
                converted_tools.append(item)
            body["tools"] = converted_tools
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                body["tool_choice"] = {"type": "tool", "name": (tool_choice.get("function") or {}).get("name")}
            elif tool_choice in (None, "auto", "required", "none"):
                body["tool_choice"] = {"type": {"required": "any"}.get(tool_choice, tool_choice or "auto")}
            else:
                raise AnyIntError("Unsupported Anthropic tool choice")
            if parallel_tool_calls is not None:
                body["tool_choice"]["disable_parallel_tool_use"] = not parallel_tool_calls
        return body

    async def chat_completion(self, model, messages, **kwargs):
        body = self._body(model, messages, **kwargs)
        started = time.monotonic()
        response = await self._request("POST", self.base_url + "/messages", json=body)
        if "temperature" in body and self._rejects_temperature(response):
            self._temperature_unsupported_models.add(model)
            body.pop("temperature", None)
            response = await self._request("POST", self.base_url + "/messages", json=body)
        if response.status_code >= 400:
            raise AnyIntError(f"Anthropic messages request failed: status_code={response.status_code}", model=model, status_code=response.status_code)
        payload = response.json()
        content, calls = [], []
        for block in payload.get("content") or []:
            if block.get("type") == "text":
                content.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append({"id": block["id"], "type": "function", "function": {
                    "name": block["name"], "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False)}})
        message = {"role": "assistant", "content": "".join(content)}
        if calls:
            message["tool_calls"] = calls
        usage = payload.get("usage") or {}
        prompt = sum(int(usage.get(key) or 0) for key in ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"])
        completion = int(usage.get("output_tokens") or 0)
        return {"id": payload.get("id"), "object": "chat.completion", "model": payload.get("model", model),
                "choices": [{"index": 0, "message": message, "finish_reason": {"max_tokens": "length", "tool_use": "tool_calls"}.get(payload.get("stop_reason"), "stop")}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
                "_x1_latency_ms": int((time.monotonic() - started) * 1000)}

    async def stream_chat_content(self, model, messages, **kwargs):
        body = self._body(model, messages, **kwargs)
        body["stream"] = True
        client = await self._get_client()
        async with client.stream("POST", self.base_url + "/messages", headers=self._headers(), json=body) as response:
            if response.status_code >= 400:
                raise AnyIntError(f"Anthropic stream failed: status_code={response.status_code}", model=model, status_code=response.status_code)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                if event.get("type") == "error":
                    raise AnyIntError("Anthropic stream returned an error", model=model)
                if event.get("type") == "message_stop":
                    return
                delta = event.get("delta") or {}
                if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                    yield delta.get("text", "")
        raise AnyIntError("Anthropic stream ended before message_stop", model=model)

    async def responses_create(self, model, input_value, *, instructions=None, max_output_tokens=None,
                               temperature=None, top_p=None, tools=None, tool_choice=None,
                               parallel_tool_calls=None, **extra):
        from .models import ResponsesRequest
        from .responses import build_response_object, response_input_to_messages
        if any(value is not None for value in extra.values()):
            raise AnyIntError("Unsupported Responses feature for Anthropic transport")
        request = ResponsesRequest(model=model, input=input_value, instructions=instructions)
        converted_tools = []
        for tool in tools or []:
            if tool.get("type") != "function" or not tool.get("name"):
                raise AnyIntError("Anthropic transport supports function tools only")
            converted_tools.append({"type": "function", "function": {k: tool[k] for k in ["name", "description", "parameters"] if k in tool}})
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            tool_choice = {"type": "function", "function": {"name": tool_choice.get("name")}}
        payload = await self.chat_completion(model, [m.model_dump(exclude_none=True) for m in response_input_to_messages(request)],
            max_tokens=max_output_tokens, temperature=temperature, top_p=top_p, tools=converted_tools,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls)
        message = payload["choices"][0]["message"]
        result = build_response_object(model, message["content"], payload["usage"], message.get("tool_calls"))
        if payload["choices"][0]["finish_reason"] == "length":
            result.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        result["_x1_latency_ms"] = payload["_x1_latency_ms"]
        return result
