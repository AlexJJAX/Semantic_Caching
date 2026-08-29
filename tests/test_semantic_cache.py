from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from redis_ai_portfolio.config import PortfolioSettings
from redis_ai_portfolio.semantic_cache import (
    CacheBypassPolicy,
    CacheEntry,
    CacheOutcome,
    CachePartition,
    CachePricing,
    CacheRequest,
    CalibrationPair,
    EmbeddedPrompt,
    FalseHitGuard,
    GeneratedAnswer,
    OpenAIBackend,
    SemanticCache,
    SemanticCandidate,
    calibrate_thresholds,
    canonicalize_prompt,
    create_semantic_cache_schema,
)


class FakeBackend:
    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.embed_calls: list[str] = []
        self.generate_calls: list[str] = []

    def embed(self, prompt: str) -> EmbeddedPrompt:
        self.embed_calls.append(prompt)
        return EmbeddedPrompt(self.vectors.get(prompt, [1.0, 0.0]), input_tokens=4)

    def embed_many(self, prompts: list[str]) -> list[EmbeddedPrompt]:
        return [self.embed(prompt) for prompt in prompts]

    def generate(self, prompt: str, partition: CachePartition) -> GeneratedAnswer:
        self.generate_calls.append(prompt)
        return GeneratedAnswer(
            answer=f"generated:{prompt}",
            input_tokens=20,
            output_tokens=10,
        )


class FakeStore:
    def __init__(self) -> None:
        self.entries: dict[str, CacheEntry] = {}
        self.semantic_candidates: list[SemanticCandidate] = []
        self.search_calls = 0
        self.put_calls: list[dict] = []

    def get_exact(self, partition: CachePartition, prompt: str) -> CacheEntry | None:
        return self.entries.get(partition.entry_key("portfolio", prompt))

    def search_semantic(
        self,
        partition: CachePartition,
        vector: list[float],
        *,
        distance_threshold: float,
        candidates: int,
    ) -> list[SemanticCandidate]:
        self.search_calls += 1
        return [
            candidate
            for candidate in self.semantic_candidates
            if candidate.entry.partition_fingerprint == partition.fingerprint
            and candidate.distance <= distance_threshold
        ][:candidates]

    def put(
        self,
        partition: CachePartition,
        prompt: str,
        answer: GeneratedAnswer,
        embedding: list[float],
        *,
        ttl_seconds: int,
        invalidation_tags,
        generation_cost_usd: float,
    ) -> CacheEntry:
        key = partition.entry_key("portfolio", prompt)
        now = int(time.time())
        entry = CacheEntry(
            key=key,
            prompt=prompt,
            normalized_prompt=canonicalize_prompt(prompt),
            answer=answer.answer,
            tenant=partition.tenant,
            task=partition.task,
            model=partition.model,
            prompt_version=partition.prompt_version,
            permissions_scope=partition.permissions_scope,
            partition_fingerprint=partition.fingerprint,
            invalidation_tags=tuple(invalidation_tags),
            created_at=now,
            expires_at=now + ttl_seconds,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            generation_cost_usd=generation_cost_usd,
        )
        self.entries[key] = entry
        self.put_calls.append(
            {
                "partition": partition,
                "prompt": prompt,
                "embedding": embedding,
                "ttl_seconds": ttl_seconds,
                "invalidation_tags": tuple(invalidation_tags),
            }
        )
        return entry


def partition(**overrides) -> CachePartition:
    values = {
        "tenant": "acme",
        "task": "support",
        "model": "gpt-5.6-luna",
        "prompt_version": "support-v1",
        "permissions": ("customer",),
    }
    values.update(overrides)
    return CachePartition(**values)


def cache(store: FakeStore, backend: FakeBackend) -> SemanticCache:
    return SemanticCache(
        store,
        backend,
        pricing=CachePricing(0.20, 1.20, 0.02),
        distance_threshold=0.1,
        ttl_seconds=60,
    )


class PartitionTests(unittest.TestCase):
    def test_key_is_readable_and_digest_binds_every_partition_dimension(self) -> None:
        base = partition()
        key = base.entry_key("portfolio", "How do I reset my password?")

        self.assertRegex(key, r"^portfolio:cache:acme:support:[a-f0-9]{32}$")
        variants = [
            partition(tenant="globex"),
            partition(task="billing"),
            partition(model="alternate-model"),
            partition(prompt_version="support-v2"),
            partition(permissions=("administrator",)),
        ]
        self.assertTrue(
            all(
                variant.entry_key("portfolio", "How do I reset my password?") != key
                for variant in variants
            )
        )

    def test_permissions_are_canonicalized(self) -> None:
        left = partition(permissions=("Admin", "customer", "admin"))
        right = partition(permissions=("customer", "admin"))
        self.assertEqual(left.permissions, ("admin", "customer"))
        self.assertEqual(left.permissions_scope, right.permissions_scope)

    def test_schema_indexes_all_partition_dimensions_and_exact_vector_size(self) -> None:
        settings = PortfolioSettings(
            redis_url="redis://localhost:6379/0",
            redis_namespace="portfolio",
            openai_api_key=None,
            openai_model="gpt-5.6-luna",
            openai_embedding_model="text-embedding-3-small",
            cache_distance_threshold=0.1,
            cache_ttl_seconds=3600,
        )
        schema = create_semantic_cache_schema(settings, dimensions=512).to_dict()
        fields = {field["name"]: field for field in schema["fields"]}
        for field_name in (
            "tenant",
            "task",
            "model",
            "prompt_version",
            "permissions_scope",
        ):
            self.assertEqual(fields[field_name]["type"], "tag")
        self.assertEqual(fields["embedding"]["attrs"]["dims"], 512)


class CacheAsideTests(unittest.TestCase):
    def test_trace_callback_exposes_exact_semantic_model_and_write_operations(self) -> None:
        store = FakeStore()
        backend = FakeBackend()
        semantic_cache = cache(store, backend)
        events: list[tuple[str, str, dict]] = []

        semantic_cache.answer(
            CacheRequest("Explain Redis caching", partition()),
            trace=lambda operation, status, data: events.append(
                (operation, status, dict(data))
            ),
        )

        operations = [(operation, status) for operation, status, _ in events]
        self.assertIn(("exact_lookup", "complete"), operations)
        self.assertIn(("semantic_lookup", "complete"), operations)
        self.assertIn(("model", "complete"), operations)
        self.assertIn(("cache_write", "complete"), operations)
        self.assertLess(
            operations.index(("exact_lookup", "complete")),
            operations.index(("semantic_lookup", "complete")),
        )
        self.assertLess(
            operations.index(("semantic_lookup", "complete")),
            operations.index(("model", "complete")),
        )

    def test_cold_miss_reuses_lookup_embedding_when_storing(self) -> None:
        store = FakeStore()
        backend = FakeBackend()
        semantic_cache = cache(store, backend)

        result = semantic_cache.answer(
            CacheRequest(
                "How do I reset my password?",
                partition(),
                ttl_seconds=45,
                invalidation_tags=("help-center-v3",),
            )
        )

        self.assertEqual(result.outcome, CacheOutcome.MISS)
        self.assertEqual(backend.embed_calls, ["How do I reset my password?"])
        self.assertEqual(backend.generate_calls, ["How do I reset my password?"])
        self.assertEqual(store.put_calls[0]["embedding"], [1.0, 0.0])
        self.assertEqual(store.put_calls[0]["ttl_seconds"], 45)

    def test_exact_lookup_happens_before_embedding(self) -> None:
        store = FakeStore()
        backend = FakeBackend()
        semantic_cache = cache(store, backend)
        first = semantic_cache.answer(CacheRequest("Reset password", partition()))
        backend.embed_calls.clear()
        backend.generate_calls.clear()

        result = semantic_cache.answer(CacheRequest("  RESET   PASSWORD ", partition()))

        self.assertEqual(first.outcome, CacheOutcome.MISS)
        self.assertEqual(result.outcome, CacheOutcome.EXACT_HIT)
        self.assertEqual(backend.embed_calls, [])
        self.assertEqual(backend.generate_calls, [])
        self.assertEqual(store.search_calls, 1)

    def test_semantic_hit_returns_candidate_without_generation(self) -> None:
        store = FakeStore()
        backend = FakeBackend()
        semantic_cache = cache(store, backend)
        cold = semantic_cache.answer(CacheRequest("Reset my password", partition()))
        cached = store.entries[cold.cache_key]
        store.semantic_candidates = [SemanticCandidate(cached, distance=0.04)]
        backend.generate_calls.clear()

        result = semantic_cache.answer(
            CacheRequest("How can I reset the password?", partition())
        )

        self.assertEqual(result.outcome, CacheOutcome.SEMANTIC_HIT)
        self.assertAlmostEqual(result.similarity, 0.96)
        self.assertEqual(backend.generate_calls, [])

    def test_false_hit_guard_rejects_changed_numeric_fact(self) -> None:
        store = FakeStore()
        backend = FakeBackend()
        semantic_cache = cache(store, backend)
        cold = semantic_cache.answer(CacheRequest("Download my 2024 invoice", partition()))
        cached = store.entries[cold.cache_key]
        store.semantic_candidates = [SemanticCandidate(cached, distance=0.01)]
        backend.generate_calls.clear()

        result = semantic_cache.answer(
            CacheRequest("Download my 2025 invoice", partition())
        )

        self.assertEqual(result.outcome, CacheOutcome.MISS)
        self.assertEqual(result.guard_rejections, ("different numeric facts",))
        self.assertEqual(backend.generate_calls, ["Download my 2025 invoice"])

    def test_sensitive_volatile_and_forced_requests_neither_read_nor_write(self) -> None:
        requests = [
            CacheRequest("My API key is sk-abcdefghijklmnop", partition()),
            CacheRequest("What is the live exchange rate today?", partition()),
            CacheRequest("Explain Redis", partition(), force_miss=True),
        ]
        expected = [CacheOutcome.BYPASS, CacheOutcome.BYPASS, CacheOutcome.FORCED_MISS]

        for request, outcome in zip(requests, expected, strict=True):
            with self.subTest(outcome=outcome):
                store = FakeStore()
                backend = FakeBackend()
                result = cache(store, backend).answer(request)
                self.assertEqual(result.outcome, outcome)
                self.assertEqual(backend.embed_calls, [])
                self.assertEqual(store.search_calls, 0)
                self.assertEqual(store.put_calls, [])


class ProtectionTests(unittest.TestCase):
    def test_guard_rejects_polarity_and_action_changes(self) -> None:
        guard = FalseHitGuard()
        self.assertFalse(
            guard.evaluate("Show invoices", "Do not show invoices").accepted
        )
        self.assertFalse(
            guard.evaluate("Show my account", "Delete my account").accepted
        )

    def test_task_policy_bypasses_entire_workload(self) -> None:
        policy = CacheBypassPolicy(uncacheable_tasks=("payments",))
        decision = policy.evaluate(
            "Explain this transfer",
            partition(task="payments"),
        )
        self.assertTrue(decision.bypass)
        self.assertEqual(decision.reason, "task policy")


class MetricsTests(unittest.TestCase):
    def test_metrics_separate_consumed_and_saved_tokens_and_feedback(self) -> None:
        store = FakeStore()
        backend = FakeBackend()
        semantic_cache = cache(store, backend)
        cold = semantic_cache.answer(CacheRequest("Reset my password", partition()))
        exact = semantic_cache.answer(CacheRequest("Reset my password", partition()))
        store.semantic_candidates = [
            SemanticCandidate(store.entries[cold.cache_key], distance=0.03)
        ]
        semantic = semantic_cache.answer(
            CacheRequest("How can I reset the password?", partition())
        )
        semantic_cache.record_feedback(exact, correct=True)
        semantic_cache.record_feedback(semantic, correct=False)

        snapshot = semantic_cache.metrics.snapshot()

        self.assertEqual(snapshot.requests, 3)
        self.assertEqual(snapshot.cache_hits, 2)
        self.assertAlmostEqual(snapshot.hit_rate, 2 / 3)
        self.assertEqual(snapshot.generation_input_tokens, 20)
        self.assertEqual(snapshot.generation_output_tokens, 10)
        self.assertEqual(snapshot.generation_tokens_saved, 60)
        self.assertEqual(snapshot.embedding_tokens, 8)
        self.assertEqual(snapshot.false_hits, 1)
        self.assertEqual(snapshot.false_hit_rate, 0.5)
        self.assertGreaterEqual(snapshot.latency_p95_ms, snapshot.latency_p50_ms)
        self.assertGreater(snapshot.estimated_net_cost_savings_usd, 0)


class CalibrationTests(unittest.TestCase):
    def test_calibration_selects_highest_safe_hit_rate(self) -> None:
        vectors = {
            "reset password": [1.0, 0.0],
            "recover login": [0.995, 0.1],
            "download invoice": [1.0, 0.0],
            "billing address": [0.98, 0.2],
        }
        report = calibrate_thresholds(
            [
                CalibrationPair("reset password", "recover login", True),
                CalibrationPair("download invoice", "billing address", False),
            ],
            FakeBackend(vectors),
            thresholds=(0.01, 0.03),
            max_false_hit_rate=0.0,
        )

        self.assertEqual(report.pairs, 2)
        self.assertEqual(report.recommended.distance_threshold, 0.01)
        self.assertEqual(report.recommended.false_hit_rate, 0.0)
        self.assertEqual(report.recommended.recall, 1.0)


class OpenAIBackendTests(unittest.TestCase):
    def test_adapter_uses_configured_models_dimensions_and_usage(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0])],
            usage=SimpleNamespace(prompt_tokens=7),
        )
        client.responses.create.return_value = SimpleNamespace(
            output_text="Redis answer",
            usage=SimpleNamespace(input_tokens=11, output_tokens=5),
        )
        with patch("redis_ai_portfolio.semantic_cache.OpenAI", return_value=client):
            backend = OpenAIBackend(
                api_key="test-key",
                embedding_model="text-embedding-3-small",
                embedding_dimensions=3,
            )
            embedded = backend.embed("What is Redis?")
            generated = backend.generate("What is Redis?", partition())
            backend.close()

        self.assertEqual(embedded, EmbeddedPrompt([1.0, 0.0, 0.0], 7))
        self.assertEqual(generated, GeneratedAnswer("Redis answer", 11, 5))
        client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="What is Redis?",
            dimensions=3,
            encoding_format="float",
        )
        self.assertEqual(
            client.responses.create.call_args.kwargs["model"],
            "gpt-5.6-luna",
        )
        self.assertFalse(client.responses.create.call_args.kwargs["store"])
        client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
