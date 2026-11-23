"""CPU inference service for a self-contained, hash-verified classifier bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classification_input import INPUT_SCHEMA_VERSION
from .classifier_contract import CLASSIFIER_SCHEMA, TAXONOMY_VERSION, LABELS, MULTI_LABELS, validate_result


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class ClassifierInput(BaseModel):
    state: str = Field(min_length=1, max_length=4096)
    current_start: int = Field(ge=0, le=4096)
    input_schema_version: str = INPUT_SCHEMA_VERSION


class ClassifierEngine:
    def __init__(self, bundle: Path, *, threads: int = 4):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from .classifier_model import build_classifier

        bundle = Path(bundle).resolve()
        manifest = json.loads((bundle / "manifest.json").read_text())
        required = {"model.safetensors", "config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
        if manifest.get("schema_version") != "lite_classifier_bundle.v1" or set(manifest.get("files_sha256", {})) != required:
            raise ValueError("classifier_bundle_manifest_invalid")
        for name, digest in manifest["files_sha256"].items():
            if file_sha256(bundle / name) != digest:
                raise ValueError("classifier_bundle_hash_mismatch")
        self.config = json.loads((bundle / "config.json").read_text())
        if self.config.get("input_schema_version") != INPUT_SCHEMA_VERSION or self.config.get("taxonomy_version") != TAXONOMY_VERSION:
            raise ValueError("classifier_bundle_version_unsupported")
        if type(threads) is not int or threads < 1:
            raise ValueError("classifier_threads_invalid")
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(bundle / "tokenizer"), local_files_only=True)
        self.model = build_classifier(self.config)
        self.model.load_state_dict(load_file(str(bundle / "model.safetensors")))
        self.model.eval()
        self.weights_sha256 = manifest["files_sha256"]["model.safetensors"]
        self.source_weights_sha256 = manifest.get("source_model_weights_sha256", self.weights_sha256)
        self.version = manifest["model_name"] + ":" + self.weights_sha256[:12]
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def health(self):
        return {"status": "ok", "ready": True, "model_version": self.version,
                "weights_sha256": self.weights_sha256, "device": "cpu",
                "source_weights_sha256": self.source_weights_sha256,
                "architecture": self.config.get("architecture", "modernbert_attention_heads"),
                "probability_calibration": self.config.get("probability_calibration", "temperature_scaled")}

    def classify(self, request: ClassifierInput):
        import torch

        if request.input_schema_version != INPUT_SCHEMA_VERSION:
            raise ValueError("classifier_input_schema_unsupported")
        if not 0 <= request.current_start < len(request.state):
            raise ValueError("classifier_current_start_invalid")
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("classifier_busy")
        try:
            started = time.perf_counter()
            enc = self.tokenizer(request.state, return_tensors="pt", return_offsets_mapping=True, truncation=False)
            if enc["input_ids"].shape[1] > self.config["hyperparameters"]["max_len"]:
                raise ValueError("classifier_input_exceeds_max_len")
            offsets = enc["offset_mapping"]
            # Qwen training selects tokens by their start offset; preserve that
            # boundary rule rather than inheriting ModernBERT's end-offset rule.
            if self.config.get("architecture") == "qwen35_08b_merged_attention_heads":
                current = offsets[..., 0] >= request.current_start
            else:
                current = offsets[..., 1] > request.current_start
            current = current & (offsets[..., 1] > offsets[..., 0]) & enc["attention_mask"].bool()
            with torch.inference_mode():
                logits, multi = self.model(enc["input_ids"], enc["attention_mask"], current)
                answers = {}
                for d, labels in LABELS.items():
                    temperature = self.config["temperatures"][d]
                    values = torch.softmax(logits[d][0].float() / temperature, -1).tolist()
                    best = max(range(len(values)), key=values.__getitem__)
                    answers[d] = {"value": labels[best], "confidence": values[best], "probabilities": dict(zip(labels, values))}
                for d, labels in MULTI_LABELS.items():
                    values = torch.sigmoid(multi[d][0]).tolist()
                    answers[d] = {"value": [v for v, p in zip(labels, values) if p >= 0.5],
                                  "confidence": None, "probabilities": dict(zip(labels, values))}
            return validate_result({"schema_version": CLASSIFIER_SCHEMA, "model_version": self.version,
                "input_schema_version": INPUT_SCHEMA_VERSION, "taxonomy_version": TAXONOMY_VERSION,
                "answers": answers, "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "input_tokens": int(enc["input_ids"].shape[1])})
        finally:
            self.lock.release()


def create_app(bundle: Path | None = None, *, engine=None, threads: int = 4) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        if app.state.engine is None:
            app.state.engine = ClassifierEngine(bundle, threads=threads)
        yield

    app = FastAPI(title="Lite classifier", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health():
        if app.state.engine is None:
            raise HTTPException(503, detail="classifier_not_ready")
        return app.state.engine.health()

    @app.post("/v1/classify")
    def classify(request: ClassifierInput):
        if app.state.engine is None:
            raise HTTPException(503, detail="classifier_not_ready")
        try:
            return app.state.engine.classify(request)
        except BlockingIOError:
            raise HTTPException(503, detail="classifier_busy") from None
        except ValueError as error:
            # Codes are owned by this module; never echo tokenizer/provider exceptions.
            code = str(error)
            raise HTTPException(422, detail=code if code.startswith("classifier_") else "classifier_inference_failed") from None
        except Exception:
            raise HTTPException(500, detail="classifier_inference_failed") from None

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=os.getenv("X1_CLASSIFIER_BUNDLE"), required=not bool(os.getenv("X1_CLASSIFIER_BUNDLE")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--threads", type=int, default=int(os.getenv("X1_CLASSIFIER_THREADS", "4")))
    args = parser.parse_args()
    uvicorn.run(create_app(args.bundle, threads=args.threads), host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
