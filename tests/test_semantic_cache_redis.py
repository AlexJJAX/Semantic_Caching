from __future__ import annotations

import unittest
import uuid
from dataclasses import replace

from redis.exceptions import RedisError, ResponseError

from redis_ai_portfolio.config import PortfolioSettings
from redis_ai_portfolio.redis import create_redis_client
from redis_ai_portfolio.semantic_cache import (
    CacheOutcome,
    CachePartition,
    CachePricing,
    CacheRequest,
    EmbeddedPrompt,
    GeneratedAnswer,
    RedisSemanticCacheStore,
    SemanticCache,
)


class DeterministicBackend:
    def __init__(self) -> None:
        self.embed_calls: list[str] = []
        self.generate_calls: list[str] = []

    def embed(self, prompt: str) -> EmbeddedPrompt:
        self.embed_calls.append(prompt)
        normalized = prompt.casefold()
        if "beta" in normalized:
            vector = [0.0, 1.0, 0.0]
        elif "gamma" in normalized:
            vector = [0.0, 0.0, 1.0]
        else:
            vector = [1.0, 0.0, 0.0]
        return EmbeddedPrompt(vector, input_tokens=5)

    def generate(self, prompt: str, partition: CachePartition) -> GeneratedAnswer:
        self.generate_calls.append(prompt)
        return GeneratedAnswer(f"answer:{prompt}", input_tokens=25, output_tokens=8)


class RedisSemanticCacheIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_settings = PortfolioSettings.from_env()
        cls.client = create_redis_client(cls.base_settings.redis_url)
        try:
            cls.client.ping()
            if not cls.client.execute_command("COMMAND", "INFO", "FT.CREATE"):
                raise unittest.SkipTest("Redis Search is unavailable")
            if not cls.client.execute_command("COMMAND", "INFO", "JSON.SET"):
                raise unittest.SkipTest("Redis JSON is unavailable")
        except RedisError as exc:
            cls.client.close()
            raise unittest.SkipTest(f"Redis integration unavailable: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def create_cache(self):
        settings = replace(
            self.base_settings,
            redis_namespace=f"phase3-cache-test:{uuid.uuid4()}",
            cache_distance_threshold=0.1,
            cache_ttl_seconds=120,
        )
        backend = DeterministicBackend()
        store = RedisSemanticCacheStore(settings, self.client, dimensions=3)
        semantic_cache = SemanticCache(
            store,
            backend,
            pricing=CachePricing(0.20, 1.20, 0.02),
            distance_threshold=settings.cache_distance_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        return settings, store, backend, semantic_cache

    def cleanup(self, store: RedisSemanticCacheStore) -> None:
        try:
            self.client.execute_command("FT.DROPINDEX", store.index_name, "DD")
        except ResponseError:
            pass

    def test_exact_semantic_partition_and_ttl_behavior(self) -> None:
        settings, store, backend, semantic_cache = self.create_cache()
        customer = CachePartition(
            tenant="acme",
            task="support",
            model="gpt-5.6-luna",
            prompt_version="support-v1",
            permissions=("customer",),
        )
        try:
            cold = semantic_cache.answer(
                CacheRequest("Explain alpha caching", customer, invalidation_tags=("kb-v1",))
            )
            self.assertEqual(cold.outcome, CacheOutcome.MISS)
            self.assertRegex(
                cold.cache_key or "",
                rf"^{settings.redis_namespace}:cache:acme:support:[a-f0-9]{{32}}$",
            )
            self.assertGreater(self.client.ttl(cold.cache_key), 0)
            self.assertLessEqual(self.client.ttl(cold.cache_key), 120)

            embedding_calls = len(backend.embed_calls)
            exact = semantic_cache.answer(CacheRequest("  EXPLAIN  ALPHA CACHING ", customer))
            self.assertEqual(exact.outcome, CacheOutcome.EXACT_HIT)
            self.assertEqual(len(backend.embed_calls), embedding_calls)

            semantic = semantic_cache.answer(
                CacheRequest("Describe alpha cache behavior", customer)
            )
            self.assertEqual(semantic.outcome, CacheOutcome.SEMANTIC_HIT)
            self.assertAlmostEqual(semantic.similarity or 0, 1.0)

            administrator = CachePartition(
                tenant="acme",
                task="support",
                model="gpt-5.6-luna",
                prompt_version="support-v1",
                permissions=("administrator",),
            )
            isolated = semantic_cache.answer(
                CacheRequest("Describe alpha cache behavior", administrator)
            )
            self.assertEqual(isolated.outcome, CacheOutcome.MISS)
            self.assertNotEqual(cold.cache_key, isolated.cache_key)
        finally:
            self.cleanup(store)

    def test_tag_partition_and_task_invalidation_are_scoped(self) -> None:
        _, store, _, semantic_cache = self.create_cache()
        partition = CachePartition(
            tenant="acme",
            task="support",
            model="gpt-5.6-luna",
            prompt_version="support-v1",
            permissions=("customer",),
        )
        other_permissions = CachePartition(
            tenant="acme",
            task="support",
            model="gpt-5.6-luna",
            prompt_version="support-v1",
            permissions=("administrator",),
        )
        try:
            alpha = semantic_cache.answer(
                CacheRequest("Explain alpha", partition, invalidation_tags=("kb-v1",))
            )
            beta = semantic_cache.answer(
                CacheRequest("Explain beta", partition, invalidation_tags=("kb-v1",))
            )
            gamma = semantic_cache.answer(
                CacheRequest("Explain gamma", partition, invalidation_tags=("kb-v2",))
            )
            admin = semantic_cache.answer(
                CacheRequest("Explain alpha", other_permissions, invalidation_tags=("kb-v1",))
            )

            self.assertEqual(store.invalidate_tag(partition, "kb-v1"), 2)
            self.assertFalse(self.client.exists(alpha.cache_key))
            self.assertFalse(self.client.exists(beta.cache_key))
            self.assertTrue(self.client.exists(gamma.cache_key))
            self.assertTrue(self.client.exists(admin.cache_key))

            self.assertEqual(store.invalidate_partition(partition), 1)
            self.assertFalse(self.client.exists(gamma.cache_key))
            self.assertTrue(self.client.exists(admin.cache_key))

            self.assertEqual(store.invalidate_task("acme", "support"), 1)
            self.assertFalse(self.client.exists(admin.cache_key))
        finally:
            self.cleanup(store)


if __name__ == "__main__":
    unittest.main()
