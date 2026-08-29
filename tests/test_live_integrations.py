from __future__ import annotations

import os
import unittest
import uuid
from dataclasses import replace

from redis.exceptions import RedisError, ResponseError

from agentic.Flex_rag.Langgraph_redis_agentic_flex_rag import (
    SOURCE_URLS,
    load_source_pages,
)
from redis_ai_portfolio.config import PortfolioSettings
from redis_ai_portfolio.redis import create_redis_client
from redis_ai_portfolio.semantic_cache import (
    CacheOutcome,
    CachePartition,
    CachePricing,
    CacheRequest,
    OpenAIBackend,
    RedisSemanticCacheStore,
    SemanticCache,
)

TRUTHY = {"1", "true", "yes", "on"}


def live_test_enabled(specific_flag: str) -> bool:
    """Require an explicit opt-in before contacting an external service."""
    return (
        os.getenv("RUN_LIVE_INTEGRATIONS", "").strip().casefold() in TRUTHY
        or os.getenv(specific_flag, "").strip().casefold() in TRUTHY
    )


class CountingOpenAIBackend(OpenAIBackend):
    """Expose production-adapter call counts without replacing either API."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embed_calls = 0
        self.generate_calls = 0

    def embed(self, prompt: str):
        self.embed_calls += 1
        return super().embed(prompt)

    def generate(self, prompt: str, partition: CachePartition):
        self.generate_calls += 1
        return super().generate(prompt, partition)


@unittest.skipUnless(
    live_test_enabled("RUN_LIVE_WEB_TESTS"),
    "set RUN_LIVE_WEB_TESTS=1 to contact the Flex RAG source websites",
)
class LiveSourceWebsiteTests(unittest.TestCase):
    def test_all_configured_flex_rag_sources_are_retrievable(self) -> None:
        for source_url in SOURCE_URLS:
            with self.subTest(source_url=source_url):
                pages = load_source_pages((source_url,))
                self.assertTrue(pages)
                self.assertGreater(sum(len(page.page_content) for page in pages), 1_000)
                self.assertTrue(
                    any(page.metadata.get("source") == source_url for page in pages)
                )


@unittest.skipUnless(
    live_test_enabled("RUN_LIVE_OPENAI_TESTS"),
    "set RUN_LIVE_OPENAI_TESTS=1 to spend one bounded OpenAI request",
)
class LiveOpenAIRedisTests(unittest.TestCase):
    """Prove the OpenAI adapter and Redis cache boundary in one bounded flow."""

    def test_cold_openai_response_is_persisted_and_exact_hit_avoids_api(self) -> None:
        base_settings = PortfolioSettings.from_env()
        if not base_settings.openai_api_key:
            self.fail("RUN_LIVE_OPENAI_TESTS requires OPENAI_API_KEY")

        settings = replace(
            base_settings,
            redis_namespace=f"live-integration:{uuid.uuid4().hex}",
            cache_ttl_seconds=60,
        )
        client = create_redis_client(settings.redis_url)
        backend = CountingOpenAIBackend(
            api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=512,
            max_output_tokens=96,
        )
        store = None
        try:
            client.ping()
            if not client.execute_command("COMMAND", "INFO", "FT.CREATE"):
                self.fail("Live integration tests require Redis Search")
            if not client.execute_command("COMMAND", "INFO", "JSON.SET"):
                self.fail("Live integration tests require Redis JSON")

            store = RedisSemanticCacheStore(settings, client, dimensions=512)
            cache = SemanticCache(
                store,
                backend,
                pricing=CachePricing.from_settings(settings),
                distance_threshold=settings.cache_distance_threshold,
                ttl_seconds=settings.cache_ttl_seconds,
            )
            partition = CachePartition(
                tenant="live-test",
                task="smoke-test",
                model=settings.openai_model,
                prompt_version="v1",
                permissions=("tester",),
            )
            request = CacheRequest(
                "In one sentence, explain what a semantic cache does.",
                partition,
            )

            cold = cache.answer(request)
            exact = cache.answer(request)

            self.assertEqual(cold.outcome, CacheOutcome.MISS)
            self.assertTrue(cold.answer.strip())
            self.assertGreater(cold.generation_input_tokens, 0)
            self.assertGreater(cold.generation_output_tokens, 0)
            self.assertEqual(exact.outcome, CacheOutcome.EXACT_HIT)
            self.assertEqual(exact.answer, cold.answer)
            self.assertEqual(exact.embedding_tokens, 0)
            self.assertEqual(backend.embed_calls, 1)
            self.assertEqual(backend.generate_calls, 1)
            self.assertGreater(client.ttl(cold.cache_key), 0)
            self.assertIsNotNone(client.json().get(cold.cache_key))
        except RedisError as exc:
            self.fail(f"Live Redis integration failed: {exc}")
        finally:
            if store is not None:
                try:
                    client.execute_command("FT.DROPINDEX", store.index_name, "DD")
                except ResponseError:
                    pass
            backend.close()
            client.close()


if __name__ == "__main__":
    unittest.main()
