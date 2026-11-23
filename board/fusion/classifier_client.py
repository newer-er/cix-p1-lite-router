"""Gateway-side fixed-head backend. Does not import torch or transformers."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import httpx

from .classifier_contract import LABELS, MULTI_LABELS, validate_result
from .decision_backend import DecisionAnswer, DecisionResult, decision_input, _validate_questions


class MmbertBackend:
    name = "mmbert"

    def __init__(self, base_url: str, *, timeout: float = 60, health_timeout: float = 2, client=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.health_timeout = health_timeout
        self._client = client
        self._owned = client is None
        self._unavailable_until = 0.0

    @property
    def available(self):
        return time.monotonic() >= self._unavailable_until

    def _http(self):
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, trust_env=False)
        return self._client

    def close(self):
        if self._client is not None and self._owned:
            self._client.close()

    def health(self):
        try:
            response = self._http().get(self.base_url + "/health", timeout=min(self.timeout, self.health_timeout))
            response.raise_for_status()
            body = response.json()
            return bool(body.get("ready"))
        except Exception:
            return False

    def classify(self, messages) -> DecisionResult:
        started = time.perf_counter()
        if not self.available:
            return DecisionResult(self.name, {}, 0.0, error="backend_unavailable")
        try:
            state = decision_input(messages)
            response = self._http().post(self.base_url + "/v1/classify", json={
                "state": state.text, "current_start": len(state.text) - len(state.current_text),
                "input_schema_version": state.schema_version})
            response.raise_for_status()
        except Exception:
            # Overload and invalid inputs are per-request; do not poison healthy requests.
            if "response" not in locals() or response.status_code not in (422, 429, 503):
                self._unavailable_until = time.monotonic() + 30
            return DecisionResult(self.name, {}, (time.perf_counter() - started) * 1000, error="backend_unavailable")
        try:
            body = validate_result(response.json())
            answers = {d: DecisionAnswer(d, "choice", value=a["value"], confidence=a["confidence"],
                                         probabilities=a["probabilities"])
                       for d, a in body["answers"].items() if d in LABELS}
            return DecisionResult(self.name, answers, (time.perf_counter() - started) * 1000,
                                  model_version=body["model_version"], taxonomy_version=body["taxonomy_version"])
        except Exception:
            return DecisionResult(self.name, {}, (time.perf_counter() - started) * 1000, error="classifier_response_invalid")

    def decide(self, messages, questions):
        error = _validate_questions(questions)
        if error:
            return DecisionResult(self.name, {}, 0.0, error=error)
        result = self.classify(messages)
        if result.error:
            return result
        answers = {}
        for question in questions:
            answer = result.answers.get(question.id)
            if answer is None or question.id in MULTI_LABELS:
                answers[question.id] = DecisionAnswer.abstain(question.id, question.type, "unsupported_question")
            elif question.type != "choice":
                answers[question.id] = DecisionAnswer.abstain(question.id, question.type, "unsupported_primitive")
            elif answer.value not in question.option_names():
                answers[question.id] = DecisionAnswer.abstain(question.id, question.type, "value_not_in_options")
            else:
                answers[question.id] = answer
        return replace(result, answers=answers)
