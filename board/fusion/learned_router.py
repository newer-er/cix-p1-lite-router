from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


FEATURE_KEYS = [
    "task_type",
    "has_code_signal",
    "has_math_signal",
    "has_security_signal",
    "has_research_signal",
    "image_block_count",
    "tool_count",
]


@dataclass(frozen=True)
class LearnedRouterPrediction:
    worker_model: str
    confidence: float
    matched_key: str


class LearnedRouter:
    def __init__(self, artifact: Dict[str, Any]) -> None:
        if artifact.get("kind") != "x1.learned_router":
            raise ValueError("Invalid learned router artifact kind")
        if int(artifact.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported learned router artifact schema")
        self.artifact = artifact
        self.routes: Dict[str, Dict[str, Any]] = dict(artifact.get("routes") or {})
        self.default_worker: Optional[str] = artifact.get("default_worker")

    @classmethod
    def load(cls, path: str) -> "LearnedRouter":
        return cls(json.loads(Path(path).read_text()))

    def predict(self, features: Dict[str, Any], available_models: Iterable[str]) -> Optional[LearnedRouterPrediction]:
        available = set(available_models)
        for key in candidate_keys(features):
            route = self.routes.get(key)
            if not route:
                continue
            worker = route.get("best_worker") or route.get("best_model")
            if worker in available:
                return LearnedRouterPrediction(
                    worker_model=str(worker),
                    confidence=float(route.get("confidence", 0.0)),
                    matched_key=key,
                )
        if self.default_worker in available:
            return LearnedRouterPrediction(worker_model=str(self.default_worker), confidence=0.0, matched_key="default")
        return None


def feature_key(features: Dict[str, Any]) -> str:
    parts = []
    for key in FEATURE_KEYS:
        value = features.get(key)
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{key}={value}")
    return "|".join(parts)


def candidate_keys(features: Dict[str, Any]) -> List[str]:
    keys = [feature_key(features)]
    task_only = {"task_type": features.get("task_type")}
    keys.append(feature_key(task_only))
    return keys


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def train_router(
    labels: Iterable[Dict[str, Any]],
    *,
    source_path: Optional[str] = None,
    candidate_workers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    task_grouped: Dict[str, Counter] = defaultdict(Counter)
    all_workers: Counter = Counter()
    total = 0
    for label in labels:
        worker = label.get("best_worker") or label.get("best_model")
        features = label.get("features") or {}
        if not worker or not features:
            continue
        worker = str(worker)
        grouped[feature_key(features)][worker] += 1
        task_grouped[feature_key({"task_type": features.get("task_type")})][worker] += 1
        all_workers[worker] += 1
        total += 1

    routes: Dict[str, Dict[str, Any]] = {}
    for key, counts in {**task_grouped, **grouped}.items():
        worker, count = _winner(counts)
        routes[key] = {
            "best_worker": worker,
            "support": sum(counts.values()),
            "confidence": count / max(1, sum(counts.values())),
            "counts": dict(sorted(counts.items())),
        }

    default_worker = _winner(all_workers)[0] if all_workers else None
    return {
        "schema_version": 1,
        "kind": "x1.learned_router",
        "artifact_id": f"lr_{uuid.uuid4().hex}",
        "created_at": int(time.time()),
        "feature_schema_version": 1,
        "limitation": "Single-call worker router trained from offline eval labels; not TRINITY, not Conductor, not workflow generation.",
        "provenance": {
            "source_path": source_path,
            "candidate_workers": sorted(candidate_workers or list(all_workers.keys())),
        },
        "training_examples": total,
        "feature_keys": FEATURE_KEYS,
        "default_worker": default_worker,
        "routes": dict(sorted(routes.items())),
    }


def _winner(counts: Counter) -> Tuple[Optional[str], int]:
    if not counts:
        return None, 0
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="routing_labels.jsonl path")
    parser.add_argument("--output", required=True, help="learned router JSON artifact path")
    args = parser.parse_args()
    artifact = train_router(read_jsonl(Path(args.labels)), source_path=args.labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output), "training_examples": artifact["training_examples"], "routes": len(artifact["routes"])}, indent=2))


if __name__ == "__main__":
    main()
