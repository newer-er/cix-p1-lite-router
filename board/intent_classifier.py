"""Orange Pi demo classifier using archived 0.6B intent and live M47b heads."""

import argparse
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from safetensors.torch import load_file
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from fusion.classification_input import INPUT_SCHEMA_VERSION
from fusion.classifier_contract import LABELS, validate_result
from fusion.classifier_serve import ClassifierInput, file_sha256


INTENT_DIRECTORY = "models/embedding_intent_06b_final_20261116"
BACKBONE_DIRECTORY = "backbone/Qwen3-Embedding-0.6B"
INTENT_SHA = "9cb0dc7163202eae29e4d2afc82228a55baf7373652af50bd96bd6fa9d052b20"
INTENT_VERSION = f"intent06b_20261116:{INTENT_SHA[:12]}"
BASE_URL = "http://127.0.0.1:8811"


def require_sha(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"Model file checksum mismatch: {path.name}")


class IntentModel:
    def __init__(self, assets: Path, threads: int):
        classifier = assets / INTENT_DIRECTORY
        backbone = assets / BACKBONE_DIRECTORY
        policy_path = classifier / "training_policy.json"
        spec_path = classifier / "model_spec.json"
        require_sha(policy_path, "519294370217d33e8c615e354029bca2e4ddb60aad4c3adc1cd3140ebec9befa")
        require_sha(spec_path, "a8db68026f0dbb24687c9c299fe487b2076432acfeca78e4593cc6b9f025cec6")
        self.spec = json.loads(spec_path.read_text(encoding="utf-8"))
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
            require_sha(backbone / name, policy["model_files_sha256"][name])
        weights = classifier / "best_trainable.safetensors"
        require_sha(weights, INTENT_SHA)
        if self.spec["intent_labels"] != LABELS["intent_primary"]:
            raise ValueError("Intent label order does not match the router")

        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone, local_files_only=True, trust_remote_code=False, padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = AutoModel.from_pretrained(
            backbone, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
            attn_implementation="sdpa",
        ).eval().requires_grad_(False)

        state = load_file(str(weights))
        self.head_weight = state.pop("head.weight")
        self.head_bias = state.pop("head.bias")
        self.merge_adapter(state, self.spec["lora"])
        hidden_size = self.encoder.config.hidden_size
        label_count = len(LABELS["intent_primary"])
        if self.head_weight.shape != (label_count, 2 * hidden_size) or self.head_bias.shape != (label_count,):
            raise ValueError("Unexpected intent head dimensions")

    def merge_adapter(self, state: dict, lora: dict) -> None:
        targets = set(lora["lora_targets"])
        rank = lora["lora_rank"]
        scale = lora["lora_alpha"] / rank
        with torch.no_grad():
            for name, layer in self.encoder.named_modules():
                if not isinstance(layer, torch.nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
                    continue
                prefix = f"encoder.base_model.model.{name}"
                a = state.pop(f"{prefix}.lora_A.default.weight")
                b = state.pop(f"{prefix}.lora_B.default.weight")
                if a.shape != (rank, layer.in_features) or b.shape != (layer.out_features, rank):
                    raise ValueError(f"Unexpected adapter dimensions: {name}")
                layer.weight.addmm_(b, a, alpha=scale)
        if state:
            raise ValueError("Unmapped LoRA tensors")

    def predict(self, request: ClassifierInput) -> dict:
        instructions = self.spec["instructions"]
        current = request.state[request.current_start:]
        prompts = [
            f"Instruct: {instructions['current']}\nQuery:{current}",
            f"Instruct: {instructions['context']}\nQuery:{request.state}",
        ]
        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        if tokens["input_ids"].shape[-1] > self.spec["max_tokens"]:
            raise HTTPException(422, detail="classifier_input_exceeds_max_len")
        with torch.inference_mode():
            hidden = self.encoder(**tokens, use_cache=False).last_hidden_state
            pooled = F.normalize(hidden[:, -1, :].float(), p=2, dim=-1)
            logits = F.linear(pooled.reshape(1, -1), self.head_weight, self.head_bias)
            values = logits.float().softmax(-1)[0].tolist()
        labels = LABELS["intent_primary"]
        best = max(range(len(labels)), key=values.__getitem__)
        return {"value": labels[best], "confidence": values[best], "probabilities": dict(zip(labels, values))}


def combined_version(base_version: str) -> str:
    if not base_version.startswith("m47b_"):
        raise ValueError("Expected the M47b classifier")
    return f"{INTENT_VERSION}+{base_version}"


def create_app(assets: Path, threads: int, base_url: str = BASE_URL) -> FastAPI:
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.intent = IntentModel(assets, threads)
        with httpx.Client(base_url=base_url, timeout=45, trust_env=False) as base:
            app.state.base = base
            yield

    app = FastAPI(title="Lite Router board intent profile", lifespan=lifespan)

    @app.get("/health")
    def health():
        try:
            response = app.state.base.get("/health", timeout=1)
            response.raise_for_status()
            status = response.json()
            if status.get("ready") is not True:
                return {"status": "degraded", "ready": False}
            version = combined_version(status["model_version"])
        except (httpx.HTTPError, KeyError, ValueError):
            return {"status": "degraded", "ready": False}
        return {"status": "ok", "ready": True, "model_version": version, "device": "cpu"}

    @app.post("/v1/classify")
    def classify(request: ClassifierInput):
        if request.input_schema_version != INPUT_SCHEMA_VERSION or request.current_start >= len(request.state):
            raise HTTPException(422, detail="classifier_input_invalid")
        if not lock.acquire(blocking=False):
            raise HTTPException(503, detail="classifier_busy")
        try:
            started = time.perf_counter()
            response = app.state.base.post("/v1/classify", json=request.model_dump())
            response.raise_for_status()
            result = validate_result(response.json())
            base_version = result["model_version"]
            result["model_version"] = combined_version(base_version)
            result["answers"]["intent_primary"] = app.state.intent.predict(request)
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            result["head_models"] = dict.fromkeys(result["answers"], base_version)
            result["head_models"]["intent_primary"] = INTENT_VERSION
            return validate_result(result)
        except httpx.HTTPStatusError as error:
            status = 422 if error.response.status_code == 422 else 503
            raise HTTPException(status, detail="m47b_classification_failed") from None
        except (httpx.RequestError, ValueError):
            raise HTTPException(503, detail="classifier_unavailable") from None
        finally:
            lock.release()

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8824)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--base-classifier-url", default=BASE_URL)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.assets, args.threads, args.base_classifier_url),
        host="127.0.0.1", port=args.port, access_log=False,
    )
