"""Decision contracts and baseline backends for the Lite routing layer.

The router currently makes decisions in three different ways: keyword matching in
``router.classify_task``, a keyword rule for the work strategy, and an optional
0.5B embedding head (``coordinator.serve.TrinityRouter``). This module puts those
behind one small, typed interface so a decision source can be swapped, evaluated
side by side, and rolled back by changing one environment variable.

The contract mirrors TypeSafe's System One request shape (``state`` plus a map of
typed questions with ``choice`` / ``noul`` / ``score`` primitives) so a hosted or
self-hosted System One implementation can later implement the same interface. The
question ``id`` is chosen by the caller and is never shown to a model; answers come
back keyed by that same id.

Only the two *baseline* backends live here. They wrap existing behaviour and are
expected to be beaten by a trained model later; nothing in this module changes the
serving path on its own.

Two rules matter for everything built on top of this:

* ``decision_input()`` is the only supported way to build a decision state. It
  delegates to ``classification_input.serialize_messages`` so that offline
  evaluation, Teacher labelling, and (later) model training all see byte-identical
  input. A backend that assembles its own text would silently break that.
* A backend that cannot answer says so. It returns an abstention rather than a
  plausible-looking guess, because a guess is indistinguishable from a real
  prediction in every metric downstream.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from .classification_input import INPUT_SCHEMA_VERSION, ClassificationInput, serialize_messages
from .models import ChatMessage

DECISION_CONTRACT_VERSION = "decision_contract.v1"

# The three System One question primitives. Each asks a different kind of question
# and returns a different kind of answer.
PRIMITIVES = ("choice", "noul", "score")

# Question ids this contract reserves. A backend recognises a question by its id so
# that evaluation harnesses and future model backends can share one naming scheme.
QUESTION_DOMAIN = "domain"
QUESTION_STRATEGY = "strategy"
QUESTION_WORKER = "worker_choice"
KNOWN_QUESTION_IDS = frozenset({QUESTION_DOMAIN, QUESTION_STRATEGY, QUESTION_WORKER})

# Answers a heuristic cannot produce, and the reason it cannot. Kept as constants so
# evaluation can group abstentions without string-matching in several places.
ABSTAIN_UNSUPPORTED_QUESTION = "unsupported_question"
ABSTAIN_UNSUPPORTED_PRIMITIVE = "unsupported_primitive"
ABSTAIN_VALUE_NOT_IN_OPTIONS = "value_not_in_options"
ABSTAIN_NO_PREDICTION = "no_prediction"
ABSTAIN_INVALID_CRITERIA = "invalid_criteria"

ERROR_BACKEND_UNAVAILABLE = "backend_unavailable"
ERROR_NO_QUESTIONS = "no_questions"
ERROR_INTERNAL = "internal_error"

# Work strategies the Lite pipeline can act on. ``vote`` exists upstream but is not
# part of the auto default after the WEIGHTS_STUDY re-measurement, so the baseline
# never selects it.
STRATEGIES = ("exec_guided", "static")


class DecisionContractError(ValueError):
    """A question violates the frozen contract.

    Raised at construction time only, so a malformed question is a programming error,
    never a runtime surprise from model output.
    """


@dataclass(frozen=True)
class DecisionQuestion:
    """One atomic decision to evaluate against the state.

    ``instructions`` carries the question in natural language. ``criteria`` depends on
    ``type``: a ``choice`` maps option name to an optional description, a ``score`` is
    an ordered list of level descriptions, and a ``noul`` takes no criteria (or a
    ``{"true": ..., "false": ...}`` pair of descriptions).
    """

    id: str
    type: str
    instructions: str
    criteria: Optional[Any] = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise DecisionContractError("question id must be non-empty")
        if self.type not in PRIMITIVES:
            raise DecisionContractError(
                f"unsupported question type {self.type!r}; expected one of {PRIMITIVES}"
            )
        if not self.instructions or not self.instructions.strip():
            raise DecisionContractError(f"question {self.id!r} needs non-empty instructions")
        if self.type == "choice":
            options = self.option_names()
            if len(options) < 2:
                raise DecisionContractError(
                    f"choice question {self.id!r} needs at least two options"
                )
        elif self.type == "score":
            levels = self.score_levels()
            if len(levels) < 2:
                raise DecisionContractError(
                    f"score question {self.id!r} needs at least two levels"
                )

    def option_names(self) -> List[str]:
        """The option names for a ``choice`` question, in declaration order."""
        if self.type != "choice" or not isinstance(self.criteria, dict):
            return []
        return [str(name) for name in self.criteria.keys()]

    def score_levels(self) -> List[str]:
        """The ordered level descriptions for a ``score`` question."""
        if self.type != "score" or not isinstance(self.criteria, (list, tuple)):
            return []
        return [str(level) for level in self.criteria]

    def to_request_body(self) -> Dict[str, Any]:
        """The question as it appears in a System One ``questions`` map.

        The id is deliberately absent: TypeSafe's contract states the caller-chosen key
        is not sent to the model and does not participate in inference.
        """
        body: Dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


@dataclass(frozen=True)
class DecisionAnswer:
    """One answer, or an explicit refusal to answer.

    ``abstained`` is the load-bearing field. A backend that does not actually support a
    question must set it rather than return a plausible default: a fabricated ``general``
    would be scored as a real prediction and quietly corrupt every downstream metric.
    ``confidence`` stays ``None`` when the decision source does not produce a probability
    distribution, which is different from being uncertain.
    """

    question_id: str
    type: str
    value: Any = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None

    @classmethod
    def abstain(cls, question_id: str, question_type: str, reason: str) -> "DecisionAnswer":
        return cls(
            question_id=question_id,
            type=question_type,
            abstained=True,
            abstain_reason=reason,
        )

    def to_trace(self) -> Dict[str, Any]:
        """A JSON-serialisable view for tracing and evaluation reports."""
        payload: Dict[str, Any] = {
            "question_id": self.question_id,
            "type": self.type,
            "value": self.value,
            "confidence": self.confidence,
            "abstained": self.abstained,
        }
        if self.abstain_reason:
            payload["abstain_reason"] = self.abstain_reason
        if self.probabilities:
            payload["probabilities"] = dict(self.probabilities)
        return payload


@dataclass(frozen=True)
class DecisionResult:
    """The outcome of one decision call, including the degraded ones.

    A backend never raises. A failure that prevents any answer at all is reported as
    ``error``; a failure that only affects one question is reported as an abstention on
    that question and leaves the rest of the answers intact.
    """

    backend: str
    answers: Dict[str, DecisionAnswer]
    latency_ms: float
    model_version: Optional[str] = None
    input_schema_version: str = INPUT_SCHEMA_VERSION
    taxonomy_version: Optional[str] = None
    contract_version: str = DECISION_CONTRACT_VERSION
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        """Whether this result carries at least one real (non-abstained) answer."""
        if self.error is not None:
            return False
        return any(not answer.abstained for answer in self.answers.values())

    def answer_for(self, question_id: str) -> Optional[DecisionAnswer]:
        return self.answers.get(question_id)

    def to_trace(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "contract_version": self.contract_version,
            "input_schema_version": self.input_schema_version,
            "model_version": self.model_version,
            "taxonomy_version": self.taxonomy_version,
            "latency_ms": round(self.latency_ms, 3),
            "error": self.error,
            "usable": self.usable,
            "answers": [answer.to_trace() for answer in self.answers.values()],
        }


class DecisionBackend(Protocol):
    """A source of typed decisions.

    Implementations must satisfy five rules, matching the graceful-degradation stance
    already used by ``coordinator.serve.TrinityRouter`` (a missing GPU must never take
    the service down):

    1. never propagate an exception -- catch everything and report it in the result;
    2. load lazily, on the first ``decide`` call rather than at import or construction;
    3. cache a load failure so it is not retried on every request;
    4. report ``ERROR_BACKEND_UNAVAILABLE`` rather than raising when unusable;
    5. build the state through :func:`decision_input` and nothing else.
    """

    name: str

    @property
    def available(self) -> bool: ...

    def decide(
        self,
        messages: Sequence[ChatMessage],
        questions: Sequence[DecisionQuestion],
    ) -> DecisionResult: ...


def decision_input(messages: Iterable[Any], *, max_chars: Optional[int] = None) -> ClassificationInput:
    """Build the decision state, the one supported way.

    Delegates to the frozen ``classification_input.serialize_messages`` so that every
    backend, the offline audit, the labelling pipeline, and future training data all
    see identical text. Backends must not concatenate messages themselves.
    """
    kwargs: Dict[str, Any] = {}
    if max_chars is not None:
        kwargs["max_chars"] = max_chars
    return serialize_messages(messages, **kwargs)


def _validate_questions(questions: Sequence[DecisionQuestion]) -> Optional[str]:
    """Return an error code when the question set cannot be evaluated at all."""
    if not questions:
        return ERROR_NO_QUESTIONS
    seen: set[str] = set()
    for question in questions:
        if not isinstance(question, DecisionQuestion):
            return ERROR_INTERNAL
        if question.id in seen:
            # Duplicate ids would make the answer map ambiguous.
            return ERROR_INTERNAL
        seen.add(question.id)
    return None


def _empty_result(backend: str, error: str, started: float) -> DecisionResult:
    return DecisionResult(
        backend=backend,
        answers={},
        latency_ms=(time.perf_counter() - started) * 1000.0,
        error=error,
    )


class HeuristicBackend:
    """The current keyword router, exposed through the decision contract.

    This is the baseline a trained model has to beat, so it is deliberately honest about
    its limits: it answers only the questions the keyword router can actually answer and
    abstains on everything else. In particular ``confidence`` is always ``None`` --
    ``classify_task`` returns a bare label with no uncertainty information, and inventing
    a number here would make the coverage/accuracy comparison meaningless.
    """

    name = "heuristic"

    def __init__(self, *, max_chars: Optional[int] = None) -> None:
        self._max_chars = max_chars

    @property
    def available(self) -> bool:
        return True

    def decide(
        self,
        messages: Sequence[ChatMessage],
        questions: Sequence[DecisionQuestion],
    ) -> DecisionResult:
        started = time.perf_counter()
        error = _validate_questions(questions)
        if error:
            return _empty_result(self.name, error, started)

        message_list = list(messages)
        state = decision_input(message_list, max_chars=self._max_chars)
        answers: Dict[str, DecisionAnswer] = {}
        try:
            for question in questions:
                answers[question.id] = self._answer(question, message_list, state)
        except Exception:
            return _empty_result(self.name, ERROR_INTERNAL, started)

        return DecisionResult(
            backend=self.name,
            answers=answers,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_version=None,
        )

    def _answer(
        self,
        question: DecisionQuestion,
        message_list: List[ChatMessage],
        state: ClassificationInput,
    ) -> DecisionAnswer:
        from .router import classify_task

        if question.id == QUESTION_DOMAIN:
            if question.type != "choice":
                return DecisionAnswer.abstain(
                    question.id, question.type, ABSTAIN_UNSUPPORTED_PRIMITIVE
                )
            options = question.option_names()
            if not options:
                return DecisionAnswer.abstain(
                    question.id, question.type, ABSTAIN_INVALID_CRITERIA
                )
            task_type = classify_task(message_list)
            if task_type not in options:
                # ``classify_task`` can return "multimodal", which taxonomy v2 removed
                # from its domain enum (it became the ``requires_vision`` capability
                # field). That is a capability question answered by deterministic code,
                # not a semantic domain, so the baseline refuses rather than coercing it
                # to a neighbour option.
                return DecisionAnswer.abstain(
                    question.id, question.type, ABSTAIN_VALUE_NOT_IN_OPTIONS
                )
            return DecisionAnswer(
                question_id=question.id,
                type=question.type,
                value=task_type,
                confidence=None,
            )

        if question.id == QUESTION_STRATEGY:
            if question.type != "choice":
                return DecisionAnswer.abstain(
                    question.id, question.type, ABSTAIN_UNSUPPORTED_PRIMITIVE
                )
            options = question.option_names()
            strategy = HeuristicBackend.keyword_strategy(state.text, message_list)
            if strategy not in options:
                return DecisionAnswer.abstain(
                    question.id, question.type, ABSTAIN_VALUE_NOT_IN_OPTIONS
                )
            return DecisionAnswer(
                question_id=question.id,
                type=question.type,
                value=strategy,
                confidence=None,
            )

        # noul/score questions and any other id need semantics the keyword router does
        # not have. The trained backends are expected to cover these.
        return DecisionAnswer.abstain(question.id, question.type, ABSTAIN_UNSUPPORTED_QUESTION)

    @staticmethod
    def keyword_strategy(
        text: str, messages: Optional[Sequence[ChatMessage]] = None
    ) -> str:
        """The pre-classifier fallback branch of ``workflows._auto_workflow_mode``.

        Mirrors that function's two rules: non-code tasks are always ``static``, and a
        code task only escalates to ``exec_guided`` when the request actually asks for
        code to be produced (a question merely *mentioning* code must not pay the exec
        fan-out). The 0.5B strategy classifier takes priority upstream; ``TrinityBackend``
        wraps that one.

        ``messages`` should be the caller's real message list when available. The
        task-type check then matches the serving path exactly; passing only ``text``
        falls back to classifying the serialized form, whose history section can tip the
        keyword counts.
        """
        from .router import classify_task

        if messages is None:
            messages = [ChatMessage(role="user", content=text)]
        task_type = classify_task(messages)
        if task_type not in {"coding", "debugging"}:
            return "static"
        lowered = text.lower()
        if not (_CODE_INTENT.search(lowered) and _CODE_SHAPE.search(lowered)):
            return "static"
        return "exec_guided"


# Same shapes ``workflows`` uses to decide whether a request wants code produced. Kept as
# module-level state so the check above can run on serialized text without building a
# ChatCompletionRequest.
_CODE_INTENT = re.compile(
    r"\b(write|implement|create|build|fix|debug|refactor|generate|complete|modify|patch|solve|develop|extend)\b"
)
_CODE_SHAPE = re.compile(
    r"```|\b(function|class|method|script|program|code|snippet|bug|unit test|test case|stdin|stdout|api endpoint|regex)\b"
)



class TrinityBackend:
    """The trained 0.5B coordinator, exposed through the decision contract.

    Thin wrapper: it delegates to the existing ``TrinityRouter`` rather than
    reimplementing the head scoring, so the two cannot drift apart. The heavy import
    happens on first use, and a missing checkpoint or absent torch degrades to
    ``available == False`` instead of taking the service down.
    """

    name = "trinity"

    def __init__(
        self,
        head_path: str,
        *,
        device: Optional[str] = None,
        strategy_path: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> None:
        self._head_path = head_path
        self._device = device
        self._strategy_path = strategy_path
        self._max_chars = max_chars
        self._router: Any = None
        self._failed = False

    @property
    def available(self) -> bool:
        self._ensure()
        return self._router is not None

    def _ensure(self) -> None:
        if self._router is not None or self._failed:
            return
        try:
            from .coordinator.serve import TrinityRouter

            self._router = TrinityRouter(
                self._head_path,
                device=self._device,
                strategy_path=self._strategy_path,
            )
        except Exception:
            self._failed = True
            self._router = None

    def decide(
        self,
        messages: Sequence[ChatMessage],
        questions: Sequence[DecisionQuestion],
    ) -> DecisionResult:
        started = time.perf_counter()
        error = _validate_questions(questions)
        if error:
            return _empty_result(self.name, error, started)

        self._ensure()
        if self._router is None:
            return _empty_result(self.name, ERROR_BACKEND_UNAVAILABLE, started)

        message_list = list(messages)
        answers: Dict[str, DecisionAnswer] = {}
        try:
            for question in questions:
                answers[question.id] = self._answer(question, message_list)
        except Exception:
            return _empty_result(self.name, ERROR_INTERNAL, started)

        return DecisionResult(
            backend=self.name,
            answers=answers,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_version=getattr(self._router, "model_name", None),
        )

    def _answer(
        self, question: DecisionQuestion, message_list: List[ChatMessage]
    ) -> DecisionAnswer:
        if question.type != "choice":
            return DecisionAnswer.abstain(
                question.id, question.type, ABSTAIN_UNSUPPORTED_PRIMITIVE
            )
        options = question.option_names()
        if not options:
            return DecisionAnswer.abstain(question.id, question.type, ABSTAIN_INVALID_CRITERIA)

        if question.id == QUESTION_STRATEGY:
            strategy = self._router.classify_strategy(message_list)
            if strategy is None or strategy not in options:
                return DecisionAnswer.abstain(
                    question.id, question.type, ABSTAIN_NO_PREDICTION
                )
            return DecisionAnswer(
                question_id=question.id, type=question.type, value=strategy, confidence=None
            )

        if question.id == QUESTION_WORKER:
            worker = self._router.predict_worker(message_list, options)
            if worker is None or worker not in options:
                return DecisionAnswer.abstain(
                    question.id, question.type, ABSTAIN_NO_PREDICTION
                )
            return DecisionAnswer(
                question_id=question.id, type=question.type, value=worker, confidence=None
            )

        return DecisionAnswer.abstain(question.id, question.type, ABSTAIN_UNSUPPORTED_QUESTION)


# Registry of backend names this build can construct. Ordered by the order a new
# deployment should try them: the baselines first, trained local models later.
BACKEND_NAMES = ("heuristic", "trinity", "mmbert")


def build_backend(name: str, settings: Any = None) -> Optional[DecisionBackend]:
    """Construct a backend by name, or ``None`` when the name is unknown/unusable.

    Returning ``None`` rather than raising keeps the serving path safe: an operator
    typo in ``X1_DECISION_BACKEND`` must not stop the service from starting, it should
    only mean the caller keeps using its existing heuristic.
    """
    if not name:
        return None
    normalised = str(name).strip().lower()
    try:
        if normalised == "mmbert":
            from .classifier_client import MmbertBackend
            return MmbertBackend(
                getattr(settings, "classifier_base_url", "http://127.0.0.1:8811"),
                timeout=getattr(settings, "classifier_timeout_seconds", 60.0),
                health_timeout=getattr(settings, "classifier_health_timeout_seconds", 2.0),
            )
        if normalised == "heuristic":
            return HeuristicBackend()
        if normalised == "trinity":
            head_path = getattr(settings, "trinity_router_path", None)
            if not head_path:
                return None
            return TrinityBackend(
                head_path,
                device=getattr(settings, "trinity_device", None),
            )
    except Exception:
        return None
    return None
