"""Production-shaped cache-aside semantics for the Redis AI portfolio."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from openai import OpenAI
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorRangeQuery
from redisvl.query.filter import FilterExpression, Num, Tag
from redisvl.schema import IndexSchema

from .config import PortfolioSettings

DEFAULT_EMBEDDING_DIMENSIONS = 512
DEFAULT_SEMANTIC_CANDIDATES = 5
METRICS_SAMPLE_LIMIT = 10_000
GUARD_VERSION = "facts-polarity-intent-v1"

_KEY_SEGMENT = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"(?<!\w)(?:[$£€]\s*)?\d[\d,]*(?:\.\d+)?%?(?!\w)")
_IDENTIFIER = re.compile(r"\b(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9_-]+\b")
_QUOTED = re.compile(r"['\"]([^'\"]{2,80})['\"]")
_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_API_SECRET = re.compile(
    r"\b(?:sk-[a-z0-9_-]{12,}|bearer\s+[a-z0-9._-]{12,}|"
    r"(?:password|passcode|secret|api[- ]?key|access[- ]?token)\s*(?:is|=|:)\s*\S+)",
    re.IGNORECASE,
)
_VOLATILE = re.compile(
    r"\b(?:latest|right now|currently|today|tonight|tomorrow|live|real[- ]?time|"
    r"weather|forecast|stock price|share price|exchange rate|sports? score|breaking news|"
    r"availability|inventory|traffic)\b",
    re.IGNORECASE,
)
_SIDE_EFFECT = re.compile(
    r"\b(?:cancel|delete|transfer|purchase|buy|book|submit|send|refund|change|update)\b"
    r".{0,24}\b(?:my|account|booking|order|payment|subscription|profile|address)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|don't|doesn't|didn't|isn't|aren't|cannot|can't|"
    r"non-(?:administrator|admin|member|user|customer|employee)s?)\b"
)

_INTENT_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "quantity": re.compile(r"\b(?:how many|how much|count|number|price|cost|total)\b"),
    "time": re.compile(r"\b(?:when|what date|which date|what year|which year)\b"),
    "identity": re.compile(r"\b(?:who|whose)\b"),
    "location": re.compile(r"\b(?:where|which location|what location)\b"),
}
_ACTION_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "read": re.compile(r"\b(?:find|show|explain|describe|list|view|check)\b"),
    "modify": re.compile(r"\b(?:change|update|edit|reset|replace)\b"),
    "delete": re.compile(r"\b(?:delete|remove|cancel|close)\b"),
    "create": re.compile(r"\b(?:create|open|book|order|purchase|buy)\b"),
}


def canonicalize_prompt(prompt: str) -> str:
    """Normalize inconsequential case and whitespace for direct exact lookup."""
    normalized = unicodedata.normalize("NFKC", prompt).strip().casefold()
    return _WHITESPACE.sub(" ", normalized)


def _validated_segment(name: str, value: str) -> str:
    normalized = value.strip().casefold()
    if not _KEY_SEGMENT.fullmatch(normalized):
        raise ValueError(
            f"{name} must be 1-64 lowercase letters, digits, or hyphens and "
            "cannot start or end with a hyphen"
        )
    return normalized


def _nonempty(name: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} cannot be empty")
    if len(cleaned) > 256:
        raise ValueError(f"{name} cannot exceed 256 characters")
    return cleaned


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachePartition:
    """Every dimension that must agree before one caller can reuse an answer."""

    tenant: str
    task: str
    model: str
    prompt_version: str
    permissions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant", _validated_segment("tenant", self.tenant))
        object.__setattr__(self, "task", _validated_segment("task", self.task))
        object.__setattr__(self, "model", _nonempty("model", self.model))
        object.__setattr__(
            self,
            "prompt_version",
            _nonempty("prompt_version", self.prompt_version),
        )
        canonical_permissions = tuple(
            sorted({_nonempty("permission", permission).casefold() for permission in self.permissions})
        )
        object.__setattr__(self, "permissions", canonical_permissions)

    @property
    def permissions_scope(self) -> str:
        if not self.permissions:
            return "public"
        return _sha256("\x1f".join(self.permissions))[:20]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "tenant": self.tenant,
                "task": self.task,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "permissions_scope": self.permissions_scope,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return _sha256(payload)[:24]

    def entry_digest(self, prompt: str) -> str:
        material = f"{self.fingerprint}\x1f{canonicalize_prompt(prompt)}"
        return _sha256(material)[:32]

    def entry_key(self, namespace: str, prompt: str) -> str:
        return ":".join(
            [
                namespace.strip(":").casefold(),
                "cache",
                self.tenant,
                self.task,
                self.entry_digest(prompt),
            ]
        )


@dataclass(frozen=True, slots=True)
class EmbeddedPrompt:
    vector: list[float]
    input_tokens: int


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    input_tokens: int
    output_tokens: int


class CacheBackend(Protocol):
    def embed(self, prompt: str) -> EmbeddedPrompt: ...

    def generate(self, prompt: str, partition: CachePartition) -> GeneratedAnswer: ...


CacheTraceCallback = Callable[[str, str, Mapping[str, Any]], None]


def _emit_cache_trace(
    callback: CacheTraceCallback | None,
    operation: str,
    status: str,
    **data: Any,
) -> None:
    """Publish optional cache diagnostics without coupling callers to an event system."""
    if callback is not None:
        callback(operation, status, data)


class SemanticCacheStore(Protocol):
    def get_exact(self, partition: CachePartition, prompt: str) -> CacheEntry | None: ...

    def search_semantic(
        self,
        partition: CachePartition,
        vector: list[float],
        *,
        distance_threshold: float,
        candidates: int,
    ) -> list[SemanticCandidate]: ...

    def put(
        self,
        partition: CachePartition,
        prompt: str,
        answer: GeneratedAnswer,
        embedding: list[float],
        *,
        ttl_seconds: int,
        invalidation_tags: Sequence[str],
        generation_cost_usd: float,
    ) -> CacheEntry: ...


@dataclass(frozen=True, slots=True)
class CachePricing:
    llm_input_per_million: float
    llm_output_per_million: float
    embedding_per_million: float

    @classmethod
    def from_settings(cls, settings: PortfolioSettings) -> CachePricing:
        return cls(
            llm_input_per_million=settings.cache_llm_input_cost_per_million,
            llm_output_per_million=settings.cache_llm_output_cost_per_million,
            embedding_per_million=settings.cache_embedding_cost_per_million,
        )

    def generation_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.llm_input_per_million
            + output_tokens * self.llm_output_per_million
        ) / 1_000_000

    def embedding_cost(self, input_tokens: int) -> float:
        return input_tokens * self.embedding_per_million / 1_000_000


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    prompt: str
    normalized_prompt: str
    answer: str
    tenant: str
    task: str
    model: str
    prompt_version: str
    permissions_scope: str
    partition_fingerprint: str
    invalidation_tags: tuple[str, ...]
    created_at: int
    expires_at: int
    input_tokens: int
    output_tokens: int
    generation_cost_usd: float

    @classmethod
    def from_document(cls, key: str, document: Mapping[str, Any]) -> CacheEntry:
        raw_tags = document.get("invalidation_tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        return cls(
            key=key,
            prompt=str(document["prompt"]),
            normalized_prompt=str(document["normalized_prompt"]),
            answer=str(document["answer"]),
            tenant=str(document["tenant"]),
            task=str(document["task"]),
            model=str(document["model"]),
            prompt_version=str(document["prompt_version"]),
            permissions_scope=str(document["permissions_scope"]),
            partition_fingerprint=str(document["partition_fingerprint"]),
            invalidation_tags=tuple(str(tag) for tag in raw_tags),
            created_at=int(float(document["created_at"])),
            expires_at=int(float(document["expires_at"])),
            input_tokens=int(float(document.get("input_tokens", 0))),
            output_tokens=int(float(document.get("output_tokens", 0))),
            generation_cost_usd=float(document.get("generation_cost_usd", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    entry: CacheEntry
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


def create_semantic_cache_schema(
    settings: PortfolioSettings,
    *,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> IndexSchema:
    """Create one Search index with strict partition fields and expiring JSON entries."""
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")
    return IndexSchema.from_dict(
        {
            "index": {
                "name": settings.redis_name("idx", "semantic-cache"),
                "prefix": f"{settings.redis_name('cache')}:",
                "key_separator": ":",
                "storage_type": "json",
            },
            "fields": [
                {"name": "tenant", "type": "tag"},
                {"name": "task", "type": "tag"},
                {"name": "model", "type": "tag"},
                {"name": "prompt_version", "type": "tag"},
                {"name": "permissions_scope", "type": "tag"},
                {"name": "partition_fingerprint", "type": "tag"},
                {
                    "name": "invalidation_tags",
                    "path": "$.invalidation_tags[*]",
                    "type": "tag",
                },
                {"name": "expires_at", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": dimensions,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


class RedisSemanticCacheStore:
    """Redis JSON/Search persistence with direct exact reads and filtered vector search."""

    def __init__(
        self,
        settings: PortfolioSettings,
        redis_client: Redis,
        *,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ) -> None:
        self.settings = settings
        self.redis_client = redis_client
        self.dimensions = dimensions
        self.schema = create_semantic_cache_schema(settings, dimensions=dimensions)
        self.index = SearchIndex(
            schema=self.schema,
            redis_client=redis_client,
            validate_on_load=True,
        )
        self.index.create(overwrite=False)

    @property
    def index_name(self) -> str:
        return self.schema.index.name

    def get_exact(self, partition: CachePartition, prompt: str) -> CacheEntry | None:
        key = partition.entry_key(self.settings.redis_namespace, prompt)
        document = self.redis_client.json().get(key)
        if not document:
            return None
        entry = CacheEntry.from_document(key, document)
        if entry.partition_fingerprint != partition.fingerprint:
            return None
        return entry

    def _partition_filter(self, partition: CachePartition) -> FilterExpression:
        return (
            (Tag("tenant") == partition.tenant)
            & (Tag("task") == partition.task)
            & (Tag("model") == partition.model)
            & (Tag("prompt_version") == partition.prompt_version)
            & (Tag("permissions_scope") == partition.permissions_scope)
            & (Tag("partition_fingerprint") == partition.fingerprint)
        )

    def search_semantic(
        self,
        partition: CachePartition,
        vector: list[float],
        *,
        distance_threshold: float,
        candidates: int,
    ) -> list[SemanticCandidate]:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Embedding has {len(vector)} dimensions; cache index expects {self.dimensions}"
            )
        filters = self._partition_filter(partition) & (Num("expires_at") > int(time.time()))
        results = self.index.query(
            VectorRangeQuery(
                vector=vector,
                vector_field_name="embedding",
                filter_expression=filters,
                distance_threshold=distance_threshold,
                num_results=candidates,
                return_fields=[],
                return_score=True,
                dialect=2,
            )
        )
        pipeline = self.redis_client.pipeline(transaction=False)
        for result in results:
            pipeline.json().get(result["id"])
        documents = pipeline.execute() if results else []
        semantic_candidates = []
        for result, document in zip(results, documents, strict=True):
            if not document:
                continue
            entry = CacheEntry.from_document(str(result["id"]), document)
            semantic_candidates.append(
                SemanticCandidate(entry=entry, distance=float(result["vector_distance"]))
            )
        return sorted(semantic_candidates, key=lambda candidate: candidate.distance)

    def put(
        self,
        partition: CachePartition,
        prompt: str,
        answer: GeneratedAnswer,
        embedding: list[float],
        *,
        ttl_seconds: int,
        invalidation_tags: Sequence[str],
        generation_cost_usd: float,
    ) -> CacheEntry:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1")
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Embedding has {len(embedding)} dimensions; cache index expects {self.dimensions}"
            )
        created_at = int(time.time())
        expires_at = created_at + ttl_seconds
        key = partition.entry_key(self.settings.redis_namespace, prompt)
        tags = tuple(sorted({_nonempty("invalidation tag", tag).casefold() for tag in invalidation_tags}))
        document = {
            "prompt": prompt.strip(),
            "normalized_prompt": canonicalize_prompt(prompt),
            "answer": answer.answer,
            "tenant": partition.tenant,
            "task": partition.task,
            "model": partition.model,
            "prompt_version": partition.prompt_version,
            "permissions": list(partition.permissions),
            "permissions_scope": partition.permissions_scope,
            "partition_fingerprint": partition.fingerprint,
            "invalidation_tags": list(tags),
            "guard_version": GUARD_VERSION,
            "created_at": created_at,
            "expires_at": expires_at,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "generation_cost_usd": generation_cost_usd,
            "embedding": embedding,
        }
        pipeline = self.redis_client.pipeline(transaction=True)
        pipeline.json().set(key, "$", document)
        pipeline.expire(key, ttl_seconds)
        pipeline.execute()
        return CacheEntry.from_document(key, document)

    def invalidate_prompt(self, partition: CachePartition, prompt: str) -> int:
        key = partition.entry_key(self.settings.redis_namespace, prompt)
        return int(self.redis_client.unlink(key))

    def _invalidate_filter(self, filters: FilterExpression) -> int:
        deleted = 0
        while True:
            results = self.index.query(
                FilterQuery(
                    filter_expression=filters,
                    return_fields=["partition_fingerprint"],
                    num_results=500,
                    dialect=2,
                )
            )
            keys = [str(result["id"]) for result in results]
            if not keys:
                return deleted
            deleted += int(self.redis_client.unlink(*keys))

    def invalidate_partition(self, partition: CachePartition) -> int:
        return self._invalidate_filter(self._partition_filter(partition))

    def invalidate_tag(self, partition: CachePartition, tag: str) -> int:
        clean_tag = _nonempty("invalidation tag", tag).casefold()
        return self._invalidate_filter(
            self._partition_filter(partition) & (Tag("invalidation_tags") == clean_tag)
        )

    def invalidate_task(self, tenant: str, task: str) -> int:
        clean_tenant = _validated_segment("tenant", tenant)
        clean_task = _validated_segment("task", task)
        pattern = self.settings.redis_name("cache", clean_tenant, clean_task, "*")
        deleted = 0
        batch: list[Any] = []
        for key in self.redis_client.scan_iter(match=pattern, count=500):
            batch.append(key)
            if len(batch) == 500:
                deleted += int(self.redis_client.unlink(*batch))
                batch.clear()
        if batch:
            deleted += int(self.redis_client.unlink(*batch))
        return deleted


class CacheOutcome(StrEnum):
    EXACT_HIT = "exact_hit"
    SEMANTIC_HIT = "semantic_hit"
    MISS = "miss"
    BYPASS = "bypass"
    FORCED_MISS = "forced_miss"


@dataclass(frozen=True, slots=True)
class CacheRequest:
    prompt: str
    partition: CachePartition
    ttl_seconds: int | None = None
    invalidation_tags: tuple[str, ...] = ()
    force_miss: bool = False

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if self.ttl_seconds is not None and self.ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1")


@dataclass(frozen=True, slots=True)
class GuardDecision:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BypassDecision:
    bypass: bool
    reason: str | None = None


class CacheBypassPolicy:
    """Conservatively avoid retaining secrets, volatile facts, and side effects."""

    def __init__(self, *, uncacheable_tasks: Iterable[str] = ()) -> None:
        self.uncacheable_tasks = {
            _validated_segment("uncacheable task", task) for task in uncacheable_tasks
        }

    def evaluate(self, prompt: str, partition: CachePartition) -> BypassDecision:
        if partition.task in self.uncacheable_tasks:
            return BypassDecision(True, "task policy")
        if _EMAIL.search(prompt) or _CARD.search(prompt) or _SSN.search(prompt):
            return BypassDecision(True, "sensitive identifier")
        if _API_SECRET.search(prompt):
            return BypassDecision(True, "secret-like content")
        if _VOLATILE.search(prompt):
            return BypassDecision(True, "volatile query")
        if _SIDE_EFFECT.search(prompt):
            return BypassDecision(True, "side-effecting request")
        return BypassDecision(False)


@dataclass(frozen=True, slots=True)
class _PromptGuardSignals:
    numbers: frozenset[str]
    identifiers: frozenset[str]
    quoted: frozenset[str]
    negated: bool
    intents: frozenset[str]
    actions: frozenset[str]
    token_count: int


class FalseHitGuard:
    """Reject semantically close prompts whose correctness-critical facts differ."""

    @staticmethod
    def _signals(prompt: str) -> _PromptGuardSignals:
        text = canonicalize_prompt(prompt)
        return _PromptGuardSignals(
            numbers=frozenset(_NUMBER.findall(text)),
            identifiers=frozenset(_IDENTIFIER.findall(text)),
            quoted=frozenset(match.casefold() for match in _QUOTED.findall(prompt)),
            negated=bool(_NEGATION.search(text)),
            intents=frozenset(
                name for name, pattern in _INTENT_PATTERNS.items() if pattern.search(text)
            ),
            actions=frozenset(
                name for name, pattern in _ACTION_PATTERNS.items() if pattern.search(text)
            ),
            token_count=max(1, len(text.split())),
        )

    def evaluate(self, query: str, candidate: str) -> GuardDecision:
        query_signals = self._signals(query)
        candidate_signals = self._signals(candidate)
        comparisons = (
            ("numeric facts", query_signals.numbers, candidate_signals.numbers),
            ("identifiers", query_signals.identifiers, candidate_signals.identifiers),
            ("quoted literals", query_signals.quoted, candidate_signals.quoted),
        )
        for name, query_values, candidate_values in comparisons:
            if query_values != candidate_values and (query_values or candidate_values):
                return GuardDecision(False, f"different {name}")
        if query_signals.negated != candidate_signals.negated:
            return GuardDecision(False, "different polarity")
        if (
            query_signals.intents
            and candidate_signals.intents
            and query_signals.intents != candidate_signals.intents
        ):
            return GuardDecision(False, "different question intent")
        if (
            query_signals.actions
            and candidate_signals.actions
            and query_signals.actions != candidate_signals.actions
        ):
            return GuardDecision(False, "different action intent")
        length_ratio = max(query_signals.token_count, candidate_signals.token_count) / min(
            query_signals.token_count,
            candidate_signals.token_count,
        )
        if length_ratio > 3.0:
            return GuardDecision(False, "materially different prompt length")
        return GuardDecision(True)


@dataclass(frozen=True, slots=True)
class CacheResult:
    answer: str
    outcome: CacheOutcome
    latency_ms: float
    event_id: str
    cache_key: str | None = None
    similarity: float | None = None
    bypass_reason: str | None = None
    guard_rejections: tuple[str, ...] = ()
    generation_input_tokens: int = 0
    generation_output_tokens: int = 0
    embedding_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_cache_overhead_usd: float = 0.0
    estimated_cost_saved_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class CacheMetricsSnapshot:
    requests: int
    outcomes: Mapping[str, int]
    cacheable_requests: int
    cache_hits: int
    hit_rate: float
    llm_calls: int
    guard_rejections: int
    evaluated_hits: int
    false_hits: int
    false_hit_rate: float
    generation_input_tokens: int
    generation_output_tokens: int
    generation_tokens_saved: int
    embedding_tokens: int
    estimated_cost_usd: float
    estimated_cache_overhead_usd: float
    estimated_cost_saved_usd: float
    estimated_net_cost_savings_usd: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_by_outcome: Mapping[str, Mapping[str, float]]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class CacheMetrics:
    """Bounded, process-local request metrics plus explicit hit-quality feedback."""

    def __init__(self, *, sample_limit: int = METRICS_SAMPLE_LIMIT) -> None:
        if sample_limit < 1:
            raise ValueError("sample_limit must be at least 1")
        self.sample_limit = sample_limit
        self._lock = threading.Lock()
        self._outcomes: Counter[str] = Counter()
        self._latencies: list[float] = []
        self._latencies_by_outcome: dict[str, list[float]] = defaultdict(list)
        self._llm_calls = 0
        self._guard_rejections = 0
        self._feedback: dict[str, bool] = {}
        self._generation_input_tokens = 0
        self._generation_output_tokens = 0
        self._generation_tokens_saved = 0
        self._embedding_tokens = 0
        self._estimated_cost_usd = 0.0
        self._estimated_cache_overhead_usd = 0.0
        self._estimated_cost_saved_usd = 0.0

    @staticmethod
    def _append_bounded(values: list[float], value: float, limit: int) -> None:
        values.append(value)
        if len(values) > limit:
            del values[: len(values) - limit]

    def record(self, result: CacheResult) -> None:
        with self._lock:
            outcome = result.outcome.value
            self._outcomes[outcome] += 1
            self._append_bounded(self._latencies, result.latency_ms, self.sample_limit)
            self._append_bounded(
                self._latencies_by_outcome[outcome],
                result.latency_ms,
                self.sample_limit,
            )
            if result.outcome in {
                CacheOutcome.MISS,
                CacheOutcome.BYPASS,
                CacheOutcome.FORCED_MISS,
            }:
                self._llm_calls += 1
            self._guard_rejections += len(result.guard_rejections)
            if result.outcome in {CacheOutcome.EXACT_HIT, CacheOutcome.SEMANTIC_HIT}:
                self._generation_tokens_saved += (
                    result.generation_input_tokens + result.generation_output_tokens
                )
            else:
                self._generation_input_tokens += result.generation_input_tokens
                self._generation_output_tokens += result.generation_output_tokens
            self._embedding_tokens += result.embedding_tokens
            self._estimated_cost_usd += result.estimated_cost_usd
            self._estimated_cache_overhead_usd += result.estimated_cache_overhead_usd
            self._estimated_cost_saved_usd += result.estimated_cost_saved_usd

    def record_feedback(self, result: CacheResult, *, correct: bool) -> None:
        if result.outcome not in {CacheOutcome.EXACT_HIT, CacheOutcome.SEMANTIC_HIT}:
            raise ValueError("False-hit feedback applies only to cache hits")
        with self._lock:
            self._feedback[result.event_id] = correct

    def snapshot(self) -> CacheMetricsSnapshot:
        with self._lock:
            outcomes = dict(self._outcomes)
            cache_hits = outcomes.get(CacheOutcome.EXACT_HIT.value, 0) + outcomes.get(
                CacheOutcome.SEMANTIC_HIT.value, 0
            )
            cacheable_requests = cache_hits + outcomes.get(CacheOutcome.MISS.value, 0)
            evaluated_hits = len(self._feedback)
            false_hits = sum(not correct for correct in self._feedback.values())
            by_outcome = {
                outcome: {
                    "p50_ms": _percentile(values, 0.50),
                    "p95_ms": _percentile(values, 0.95),
                }
                for outcome, values in self._latencies_by_outcome.items()
            }
            return CacheMetricsSnapshot(
                requests=sum(outcomes.values()),
                outcomes=outcomes,
                cacheable_requests=cacheable_requests,
                cache_hits=cache_hits,
                hit_rate=cache_hits / cacheable_requests if cacheable_requests else 0.0,
                llm_calls=self._llm_calls,
                guard_rejections=self._guard_rejections,
                evaluated_hits=evaluated_hits,
                false_hits=false_hits,
                false_hit_rate=false_hits / evaluated_hits if evaluated_hits else 0.0,
                generation_input_tokens=self._generation_input_tokens,
                generation_output_tokens=self._generation_output_tokens,
                generation_tokens_saved=self._generation_tokens_saved,
                embedding_tokens=self._embedding_tokens,
                estimated_cost_usd=self._estimated_cost_usd,
                estimated_cache_overhead_usd=self._estimated_cache_overhead_usd,
                estimated_cost_saved_usd=self._estimated_cost_saved_usd,
                estimated_net_cost_savings_usd=(
                    self._estimated_cost_saved_usd - self._estimated_cache_overhead_usd
                ),
                latency_p50_ms=_percentile(self._latencies, 0.50),
                latency_p95_ms=_percentile(self._latencies, 0.95),
                latency_by_outcome=by_outcome,
            )


class SemanticCache:
    """Exact-first, semantic-second cache-aside orchestration around an LLM backend."""

    def __init__(
        self,
        store: SemanticCacheStore,
        backend: CacheBackend,
        *,
        pricing: CachePricing,
        distance_threshold: float,
        ttl_seconds: int,
        semantic_candidates: int = DEFAULT_SEMANTIC_CANDIDATES,
        bypass_policy: CacheBypassPolicy | None = None,
        false_hit_guard: FalseHitGuard | None = None,
        metrics: CacheMetrics | None = None,
    ) -> None:
        if not 0.0 <= distance_threshold <= 2.0:
            raise ValueError("distance_threshold must be between 0 and 2")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1")
        if semantic_candidates < 1:
            raise ValueError("semantic_candidates must be at least 1")
        self.store = store
        self.backend = backend
        self.pricing = pricing
        self.distance_threshold = distance_threshold
        self.ttl_seconds = ttl_seconds
        self.semantic_candidates = semantic_candidates
        self.bypass_policy = bypass_policy or CacheBypassPolicy()
        self.false_hit_guard = false_hit_guard or FalseHitGuard()
        self.metrics = metrics or CacheMetrics()

    def _generated_result(
        self,
        request: CacheRequest,
        *,
        started_at: float,
        outcome: CacheOutcome,
        embedding: EmbeddedPrompt | None = None,
        bypass_reason: str | None = None,
        guard_rejections: Sequence[str] = (),
        store_result: bool,
        trace: CacheTraceCallback | None = None,
    ) -> CacheResult:
        _emit_cache_trace(trace, "model", "running", model=request.partition.model)
        model_started_at = time.perf_counter()
        generated = self.backend.generate(request.prompt, request.partition)
        if not generated.answer.strip():
            raise RuntimeError("LLM backend returned an empty answer")
        _emit_cache_trace(
            trace,
            "model",
            "complete",
            duration_ms=(time.perf_counter() - model_started_at) * 1000,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )
        generation_cost = self.pricing.generation_cost(
            generated.input_tokens,
            generated.output_tokens,
        )
        embedding_cost = self.pricing.embedding_cost(
            embedding.input_tokens if embedding else 0
        )
        cache_key = None
        if store_result:
            if embedding is None:
                _emit_cache_trace(trace, "embedding", "running")
                embedding_started_at = time.perf_counter()
                embedding = self.backend.embed(request.prompt)
                _emit_cache_trace(
                    trace,
                    "embedding",
                    "complete",
                    duration_ms=(time.perf_counter() - embedding_started_at) * 1000,
                    input_tokens=embedding.input_tokens,
                )
                embedding_cost = self.pricing.embedding_cost(embedding.input_tokens)
            _emit_cache_trace(
                trace,
                "cache_write",
                "running",
                ttl_seconds=request.ttl_seconds or self.ttl_seconds,
            )
            write_started_at = time.perf_counter()
            entry = self.store.put(
                request.partition,
                request.prompt,
                generated,
                embedding.vector,
                ttl_seconds=request.ttl_seconds or self.ttl_seconds,
                invalidation_tags=request.invalidation_tags,
                generation_cost_usd=generation_cost,
            )
            cache_key = entry.key
            _emit_cache_trace(
                trace,
                "cache_write",
                "complete",
                duration_ms=(time.perf_counter() - write_started_at) * 1000,
                cache_key=cache_key,
            )
        else:
            _emit_cache_trace(trace, "cache_write", "skipped", reason=bypass_reason)
        result = CacheResult(
            answer=generated.answer,
            outcome=outcome,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            event_id=str(uuid.uuid4()),
            cache_key=cache_key,
            bypass_reason=bypass_reason,
            guard_rejections=tuple(guard_rejections),
            generation_input_tokens=generated.input_tokens,
            generation_output_tokens=generated.output_tokens,
            embedding_tokens=embedding.input_tokens if embedding else 0,
            estimated_cost_usd=generation_cost + embedding_cost,
            estimated_cache_overhead_usd=embedding_cost,
        )
        self.metrics.record(result)
        return result

    def answer(
        self,
        request: CacheRequest,
        *,
        trace: CacheTraceCallback | None = None,
    ) -> CacheResult:
        started_at = time.perf_counter()
        if request.force_miss:
            _emit_cache_trace(trace, "exact_lookup", "skipped", reason="forced miss")
            _emit_cache_trace(trace, "semantic_lookup", "skipped", reason="forced miss")
            return self._generated_result(
                request,
                started_at=started_at,
                outcome=CacheOutcome.FORCED_MISS,
                bypass_reason="forced miss",
                store_result=False,
                trace=trace,
            )

        bypass = self.bypass_policy.evaluate(request.prompt, request.partition)
        if bypass.bypass:
            _emit_cache_trace(trace, "exact_lookup", "skipped", reason=bypass.reason)
            _emit_cache_trace(trace, "semantic_lookup", "skipped", reason=bypass.reason)
            return self._generated_result(
                request,
                started_at=started_at,
                outcome=CacheOutcome.BYPASS,
                bypass_reason=bypass.reason,
                store_result=False,
                trace=trace,
            )

        _emit_cache_trace(trace, "exact_lookup", "running")
        exact_started_at = time.perf_counter()
        exact = self.store.get_exact(request.partition, request.prompt)
        _emit_cache_trace(
            trace,
            "exact_lookup",
            "complete",
            duration_ms=(time.perf_counter() - exact_started_at) * 1000,
            hit=exact is not None,
        )
        if exact is not None:
            _emit_cache_trace(trace, "semantic_lookup", "skipped", reason="exact hit")
            _emit_cache_trace(trace, "model", "skipped", reason="exact hit")
            _emit_cache_trace(trace, "cache_write", "skipped", reason="exact hit")
            result = CacheResult(
                answer=exact.answer,
                outcome=CacheOutcome.EXACT_HIT,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                event_id=str(uuid.uuid4()),
                cache_key=exact.key,
                generation_input_tokens=exact.input_tokens,
                generation_output_tokens=exact.output_tokens,
                estimated_cost_saved_usd=exact.generation_cost_usd,
            )
            self.metrics.record(result)
            return result

        _emit_cache_trace(trace, "embedding", "running")
        embedding_started_at = time.perf_counter()
        embedding = self.backend.embed(request.prompt)
        _emit_cache_trace(
            trace,
            "embedding",
            "complete",
            duration_ms=(time.perf_counter() - embedding_started_at) * 1000,
            input_tokens=embedding.input_tokens,
        )
        _emit_cache_trace(
            trace,
            "semantic_lookup",
            "running",
            distance_threshold=self.distance_threshold,
        )
        semantic_started_at = time.perf_counter()
        candidates = self.store.search_semantic(
            request.partition,
            embedding.vector,
            distance_threshold=self.distance_threshold,
            candidates=self.semantic_candidates,
        )
        _emit_cache_trace(
            trace,
            "semantic_lookup",
            "complete",
            duration_ms=(time.perf_counter() - semantic_started_at) * 1000,
            candidates=len(candidates),
        )
        guard_rejections: list[str] = []
        for candidate in candidates:
            decision = self.false_hit_guard.evaluate(request.prompt, candidate.entry.prompt)
            if not decision.accepted:
                guard_rejections.append(decision.reason or "guard rejected candidate")
                continue
            embedding_cost = self.pricing.embedding_cost(embedding.input_tokens)
            result = CacheResult(
                answer=candidate.entry.answer,
                outcome=CacheOutcome.SEMANTIC_HIT,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                event_id=str(uuid.uuid4()),
                cache_key=candidate.entry.key,
                similarity=candidate.similarity,
                guard_rejections=tuple(guard_rejections),
                generation_input_tokens=candidate.entry.input_tokens,
                generation_output_tokens=candidate.entry.output_tokens,
                embedding_tokens=embedding.input_tokens,
                estimated_cost_usd=embedding_cost,
                estimated_cache_overhead_usd=embedding_cost,
                estimated_cost_saved_usd=candidate.entry.generation_cost_usd,
            )
            _emit_cache_trace(trace, "model", "skipped", reason="semantic hit")
            _emit_cache_trace(trace, "cache_write", "skipped", reason="semantic hit")
            self.metrics.record(result)
            return result

        return self._generated_result(
            request,
            started_at=started_at,
            outcome=CacheOutcome.MISS,
            embedding=embedding,
            guard_rejections=guard_rejections,
            store_result=True,
            trace=trace,
        )

    def record_feedback(self, result: CacheResult, *, correct: bool) -> None:
        self.metrics.record_feedback(result, correct=correct)


class OpenAIBackend:
    """Responses + Embeddings API adapter with exact usage accounting."""

    def __init__(
        self,
        *,
        api_key: str,
        embedding_model: str,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        instructions: str = (
            "Answer concisely and factually. If the request lacks required context, "
            "state what is missing rather than inventing it."
        ),
        max_output_tokens: int = 300,
    ) -> None:
        self.client = OpenAI(api_key=api_key, max_retries=2, timeout=20.0)
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.instructions = instructions
        self.max_output_tokens = max_output_tokens

    def embed(self, prompt: str) -> EmbeddedPrompt:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=prompt,
            dimensions=self.embedding_dimensions,
            encoding_format="float",
        )
        return EmbeddedPrompt(
            vector=list(response.data[0].embedding),
            input_tokens=int(response.usage.prompt_tokens),
        )

    def embed_many(self, prompts: Sequence[str]) -> list[EmbeddedPrompt]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=list(prompts),
            dimensions=self.embedding_dimensions,
            encoding_format="float",
        )
        ordered_data = sorted(response.data, key=lambda item: item.index)
        return [
            EmbeddedPrompt(
                vector=list(item.embedding),
                input_tokens=max(1, math.ceil(len(prompt) / 4)),
            )
            for prompt, item in zip(prompts, ordered_data, strict=True)
        ]

    def generate(self, prompt: str, partition: CachePartition) -> GeneratedAnswer:
        response = self.client.responses.create(
            model=partition.model,
            instructions=self.instructions,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
            reasoning={"effort": "low"},
            store=False,
        )
        usage = response.usage
        return GeneratedAnswer(
            answer=response.output_text,
            input_tokens=int(usage.input_tokens if usage else 0),
            output_tokens=int(usage.output_tokens if usage else 0),
        )

    def close(self) -> None:
        self.client.close()


@dataclass(frozen=True, slots=True)
class CalibrationPair:
    query: str
    candidate: str
    should_hit: bool
    name: str = ""


@dataclass(frozen=True, slots=True)
class ThresholdScore:
    distance_threshold: float
    similarity_threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    hit_rate: float
    false_hit_rate: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    recommended: ThresholdScore
    scores: tuple[ThresholdScore, ...]
    pairs: int
    guard_rejections: int


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Vectors must be non-empty and have matching dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine distance is undefined for a zero vector")
    similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - similarity


def calibrate_thresholds(
    pairs: Sequence[CalibrationPair],
    backend: CacheBackend,
    *,
    thresholds: Sequence[float] = (
        0.03,
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ),
    max_false_hit_rate: float = 0.01,
    false_hit_guard: FalseHitGuard | None = None,
) -> CalibrationReport:
    """Evaluate labeled prompt pairs and choose the broadest safe distance threshold."""
    if not pairs:
        raise ValueError("At least one calibration pair is required")
    if not thresholds:
        raise ValueError("At least one threshold is required")
    if not 0.0 <= max_false_hit_rate <= 1.0:
        raise ValueError("max_false_hit_rate must be between 0 and 1")
    if any(not 0.0 <= threshold <= 2.0 for threshold in thresholds):
        raise ValueError("Every distance threshold must be between 0 and 2")

    unique_prompts = list(
        dict.fromkeys(prompt for pair in pairs for prompt in (pair.query, pair.candidate))
    )
    embed_many = getattr(backend, "embed_many", None)
    if callable(embed_many):
        embedded = embed_many(unique_prompts)
    else:
        embedded = [backend.embed(prompt) for prompt in unique_prompts]
    vectors = {
        prompt: result.vector for prompt, result in zip(unique_prompts, embedded, strict=True)
    }
    guard = false_hit_guard or FalseHitGuard()
    measurements = []
    guard_rejections = 0
    for pair in pairs:
        decision = guard.evaluate(pair.query, pair.candidate)
        if not decision.accepted:
            guard_rejections += 1
        measurements.append(
            (
                pair.should_hit,
                cosine_distance(vectors[pair.query], vectors[pair.candidate]),
                decision.accepted,
            )
        )

    scores = []
    for threshold in sorted(set(thresholds)):
        true_positives = false_positives = true_negatives = false_negatives = 0
        for should_hit, distance, guard_accepted in measurements:
            predicted_hit = distance <= threshold and guard_accepted
            if predicted_hit and should_hit:
                true_positives += 1
            elif predicted_hit:
                false_positives += 1
            elif should_hit:
                false_negatives += 1
            else:
                true_negatives += 1
        predicted_hits = true_positives + false_positives
        positives = true_positives + false_negatives
        precision = true_positives / predicted_hits if predicted_hits else 1.0
        recall = true_positives / positives if positives else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(
            ThresholdScore(
                distance_threshold=threshold,
                similarity_threshold=1.0 - threshold,
                true_positives=true_positives,
                false_positives=false_positives,
                true_negatives=true_negatives,
                false_negatives=false_negatives,
                hit_rate=predicted_hits / len(measurements),
                false_hit_rate=false_positives / predicted_hits if predicted_hits else 0.0,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    safe_scores = [score for score in scores if score.false_hit_rate <= max_false_hit_rate]
    if safe_scores:
        recommended = max(
            safe_scores,
            key=lambda score: (score.hit_rate, score.f1, score.distance_threshold),
        )
    else:
        recommended = min(
            scores,
            key=lambda score: (score.false_hit_rate, -score.f1, score.distance_threshold),
        )
    return CalibrationReport(
        recommended=recommended,
        scores=tuple(scores),
        pairs=len(pairs),
        guard_rejections=guard_rejections,
    )
