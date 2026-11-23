"""Authored examples for exercising the UI without a board or cloud key."""

INSPECTION = "把以下巡检交接记录整理成两条待办：A区运行正常，B区指示灯偶尔闪烁。明早复查B区电源接头，下午把复查结果补进登记表。"
DEBUGGING = "这段采样脚本遇到空列表会报 IndexError：def latest(samples): return samples[-1]。请给出能处理空列表的修复和两条最小测试。"


def prediction(question: str, label: str, other: str, confidence: float) -> dict:
    return {
        "question_id": question, "value": label, "confidence": confidence,
        "probabilities": {label: confidence, other: 1 - confidence},
    }


def replay_chat(prompt: str, profile: str) -> dict | None:
    if prompt == INSPECTION:
        task, intent, worker = "general", "transform", "claude-haiku-5-5"
        answer = "1. 明早复查 B 区电源接头。\n2. 下午把复查结果补进登记表。"
    elif prompt == DEBUGGING:
        task, intent, worker = "debugging", "fix", "claude-opus-5-5"
        answer = (
            "为空列表提供默认结果，再测试空列表和有数据的情况。\n\n```python\n"
            "def latest(samples):\n    return samples[-1] if samples else None\n\n"
            "assert latest([]) is None\nassert latest([7]) == 7\n```"
        )
    else:
        return None

    hybrid = profile == "intent06b"
    baseline = "coding" if task == "debugging" else "general"
    used = ["domain", "intent_primary"] if task == "debugging" else ["domain"]
    classification = {
        "model_version": "intent06b_replay" if hybrid else "m47b_replay",
        "policy_version": "domain_intent_v1",
        "minimum_confidence": 0.7,
        "baseline_task_type": baseline,
        "task_type": task,
        "applied": True,
        "used_dimensions": used,
        "answers": [
            prediction("domain", baseline, "math", 0.96),
            prediction("intent_primary", intent, "explain", 0.91 if hybrid else 0.86),
            prediction("requires_tools", "no", "yes", 0.98),
        ],
    }
    return {
        "demo_mode": "replay",
        "choices": [{"message": {"role": "assistant", "content": answer}}],
        "x1_route": {"worker_model": worker, "task_type": task, "workflow": {"classification": classification}},
    }
