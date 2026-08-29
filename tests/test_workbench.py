from __future__ import annotations

import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from redis.exceptions import RedisError, ResponseError

from redis_ai_portfolio.config import PortfolioSettings
from redis_ai_portfolio.redis import create_redis_client
from redis_ai_portfolio.workbench import (
    OpenAIWorkbenchBackend,
    RedisAIWorkbench,
    RunStore,
    _stable_embedding,
)


class WorkbenchUnitTests(unittest.TestCase):
    def test_run_store_replays_events_and_terminal_result(self) -> None:
        store = RunStore(max_runs=2)
        run_id = store.create("cache")
        store.emit(run_id, "prompt", "complete", "Accepted", "Sanitized prompt")
        store.complete(run_id, {"headline": "Exact hit"})

        events, terminal = store.wait_for_events(run_id, 0, timeout=0)
        snapshot = store.snapshot(run_id)

        self.assertTrue(terminal)
        self.assertEqual(events[0]["sequence"], 1)
        self.assertEqual(events[0]["stage"], "prompt")
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["result"]["headline"], "Exact hit")

    def test_local_embeddings_are_normalized_and_semantically_stable(self) -> None:
        first = _stable_embedding("Redis semantic caching reduces model latency")
        second = _stable_embedding("A Redis meaning based cache lowers LLM response latency")
        similarity = sum(left * right for left, right in zip(first, second, strict=True))

        self.assertEqual(len(first), 512)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0)
        self.assertGreater(similarity, 0.72)

    def test_live_backend_uses_responses_embeddings_and_exact_usage(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.25] * 512)],
            usage=SimpleNamespace(prompt_tokens=7),
        )
        client.responses.create.return_value = SimpleNamespace(
            output_text="A live grounded answer.",
            usage=SimpleNamespace(input_tokens=31, output_tokens=9),
        )
        with patch("redis_ai_portfolio.workbench.OpenAI", return_value=client):
            backend = OpenAIWorkbenchBackend(
                api_key="test-key",
                model="gpt-5.6-luna",
                embedding_model="text-embedding-3-small",
            )
            embedded = backend.embed("Explain Redis caching")
            generated = backend.generate_text(
                instructions="Answer from evidence.",
                input_text="Redis evidence",
                fallback_answer="unused",
            )
            backend.close()

        self.assertEqual(len(embedded.vector), 512)
        self.assertEqual(embedded.input_tokens, 7)
        self.assertEqual(generated.input_tokens, 31)
        self.assertEqual(generated.output_tokens, 9)
        self.assertEqual(generated.answer, "A live grounded answer.")
        self.assertFalse(client.responses.create.call_args.kwargs["store"])
        self.assertEqual(
            client.responses.create.call_args.kwargs["model"],
            "gpt-5.6-luna",
        )
        client.close.assert_called_once_with()

    def test_live_mode_requires_an_api_key_instead_of_silent_fallback(self) -> None:
        settings = PortfolioSettings(
            redis_url="redis://localhost:6379/0",
            redis_namespace="workbench-unit",
            openai_api_key=None,
            openai_model="gpt-5.6-luna",
            openai_embedding_model="text-embedding-3-small",
            cache_distance_threshold=0.2,
            cache_ttl_seconds=3600,
            workbench_model_mode="live",
        )
        with self.assertRaisesRegex(ValueError, "requires OPENAI_API_KEY"):
            RedisAIWorkbench(settings, MagicMock())

    def test_browser_captures_form_values_before_disabling_controls(self) -> None:
        javascript = (
            Path(__file__).resolve().parents[1] / "workbench" / "static" / "app.js"
        ).read_text(encoding="utf-8")
        submit_run = javascript[javascript.index("async function submitRun") :]

        self.assertLess(
            submit_run.index("new FormData(form)"),
            submit_run.index("setFormsDisabled(true)"),
        )


class WorkbenchRedisIntegrationTests(unittest.TestCase):
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

    def setUp(self) -> None:
        namespace = f"workbench-test:{uuid.uuid4()}"
        self.settings = replace(
            self.base_settings,
            redis_namespace=namespace,
            workbench_model_mode="demo",
        )
        self.engine = RedisAIWorkbench(
            self.settings,
            self.client,
            model_delay_seconds=0,
            event_pause_seconds=0,
        )

    def tearDown(self) -> None:
        self.engine.reset()
        self.engine.close()
        try:
            self.client.execute_command(
                "FT.DROPINDEX",
                self.settings.redis_name("idx", "semantic-cache"),
                "DD",
            )
        except ResponseError:
            pass

    def test_all_four_demos_complete_with_sanitized_redis_state(self) -> None:
        cache = self.engine.run_sync(
            "cache",
            {
                "prompt": "How does Redis semantic caching reduce LLM latency?",
                "scenario": "cold",
            },
        )
        exact = self.engine.run_sync(
            "cache",
            {"prompt": "How does Redis semantic caching reduce LLM latency?"},
        )
        memory = self.engine.run_sync(
            "memory",
            {
                "prompt": "Remember that I prefer aisle seats for morning flights.",
                "thread_id": "test-thread",
            },
        )
        recalled = self.engine.run_sync(
            "memory",
            {
                "prompt": "Which seat and flight time do I prefer?",
                "thread_id": "test-thread",
            },
        )
        rbac = self.engine.run_sync(
            "rbac",
            {"prompt": "Which quarterly reports can I export?", "role": "finance"},
        )
        cross_role = self.engine.run_sync(
            "rbac",
            {"prompt": "Which quarterly finance reports can I export?", "role": "sales"},
        )
        evaluation = self.engine.run_sync("evaluation", {})

        self.assertEqual(cache["status"], "complete")
        self.assertEqual(cache["result"]["headline"], "Miss")
        cache_write = next(
            event
            for event in cache["events"]
            if event["title"] == "TTL cache entry stored"
        )
        self.assertTrue(cache_write["detail"].startswith("cache:workbench:"))
        self.assertNotIn(self.settings.redis_namespace, cache_write["detail"])
        self.assertEqual(exact["result"]["headline"], "Exact Hit")
        self.assertEqual(memory["status"], "complete")
        self.assertEqual(memory["result"]["metrics"][0]["value"], "15 min")
        self.assertIn("aisle seats", recalled["result"]["answer"])
        self.assertEqual(recalled["result"]["metrics"][1]["value"], "no")
        ltm_keys = list(
            self.client.scan_iter(
                match=self.settings.redis_name("workbench", "ltm", "demo-user", "*"),
                count=100,
            )
        )
        self.assertEqual(len(ltm_keys), 1)
        self.assertEqual(rbac["status"], "complete")
        self.assertIn("finance-handbook.pdf", rbac["result"]["answer"])
        self.assertNotIn("finance-handbook.pdf", cross_role["result"]["answer"])
        self.assertEqual(evaluation["status"], "complete")
        self.assertEqual(evaluation["result"]["metrics"][0]["value"], "100%")

        for snapshot in (cache, memory, recalled, rbac, cross_role, evaluation):
            stages = {event["stage"] for event in snapshot["events"]}
            self.assertEqual(
                stages,
                {"prompt", "cache", "retrieval", "model", "memory", "metrics"},
            )

        inspector = self.engine.redis_inspector()
        self.assertTrue(inspector["keys"])
        self.assertTrue(inspector["indexes"])
        self.assertTrue(all(set(item) == {"key", "type", "ttl_seconds", "memory_bytes"} for item in inspector["keys"]))

    def test_sensitive_cache_prompt_is_redacted_and_not_stored(self) -> None:
        snapshot = self.engine.run_sync(
            "cache",
            {"prompt": "My API key is sk-abcdefghijklmnop"},
        )

        prompt_event = next(event for event in snapshot["events"] if event["stage"] == "prompt")
        self.assertEqual(prompt_event["detail"], "Sensitive prompt withheld")
        self.assertEqual(snapshot["result"]["headline"], "Bypass")
        self.assertFalse(
            list(
                self.client.scan_iter(
                    match=self.settings.redis_name("cache", "workbench", "*"),
                    count=100,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
