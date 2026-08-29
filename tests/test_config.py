from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from redis_ai_portfolio.config import PortfolioSettings, build_redis_url, redact_redis_url


class RedisUrlTests(unittest.TestCase):
    def test_no_password_omits_empty_authentication(self) -> None:
        self.assertEqual(
            build_redis_url(host="localhost", port=6379, database=0),
            "redis://localhost:6379/0",
        )

    def test_password_is_encoded(self) -> None:
        self.assertEqual(
            build_redis_url(
                host="redis.example.com",
                port=6380,
                database=2,
                username="app user",
                password="p@ss/word",
                ssl=True,
            ),
            "rediss://app%20user:p%40ss%2Fword@redis.example.com:6380/2",
        )

    def test_redaction_removes_credentials(self) -> None:
        self.assertEqual(
            redact_redis_url("rediss://app:secret@redis.example.com:6380/2"),
            "rediss://redis.example.com:6380/2",
        )


class SettingsTests(unittest.TestCase):
    def test_defaults_are_runnable_and_namespaced(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = PortfolioSettings.from_env(env_file="/does/not/exist")
        self.assertEqual(settings.redis_url, "redis://localhost:6379/0")
        self.assertEqual(settings.openai_model, "gpt-5.6-luna")
        self.assertEqual(settings.stm_ttl_minutes, 1440)
        self.assertTrue(settings.stm_refresh_ttl_on_read)
        self.assertEqual(settings.cache_distance_threshold, 0.2)
        self.assertEqual(settings.cache_ttl_seconds, 3600)
        self.assertEqual(settings.cache_llm_input_cost_per_million, 0.20)
        self.assertEqual(settings.cache_llm_output_cost_per_million, 1.20)
        self.assertEqual(settings.cache_embedding_cost_per_million, 0.02)
        self.assertEqual(settings.workbench_model_mode, "live")
        self.assertEqual(settings.redis_name("RBAC RAG", "Docs"), "portfolio:rbac-rag:docs")

    def test_workbench_model_mode_is_explicit_and_validated(self) -> None:
        with patch.dict(
            os.environ,
            {"WORKBENCH_MODEL_MODE": "demo"},
            clear=True,
        ):
            settings = PortfolioSettings.from_env(env_file="/does/not/exist")
        self.assertEqual(settings.workbench_model_mode, "demo")

        with patch.dict(
            os.environ,
            {"WORKBENCH_MODEL_MODE": "automatic"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be live or demo"):
                PortfolioSettings.from_env(env_file="/does/not/exist")

    def test_stm_expiry_policy_is_configurable(self) -> None:
        with patch.dict(
            os.environ,
            {
                "STM_TTL_MINUTES": "30",
                "STM_REFRESH_TTL_ON_READ": "false",
            },
            clear=True,
        ):
            settings = PortfolioSettings.from_env(env_file="/does/not/exist")
        self.assertEqual(settings.stm_ttl_minutes, 30)
        self.assertFalse(settings.stm_refresh_ttl_on_read)

    def test_blank_host_uses_local_default(self) -> None:
        with patch.dict(os.environ, {"REDIS_HOST": ""}, clear=True):
            settings = PortfolioSettings.from_env(env_file="/does/not/exist")
        self.assertEqual(settings.redis_url, "redis://localhost:6379/0")

    def test_semantic_cache_threshold_ttl_and_pricing_are_configurable(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CACHE_DISTANCE_THRESHOLD": "0.08",
                "CACHE_TTL_SECONDS": "900",
                "CACHE_LLM_INPUT_COST_PER_MILLION": "0.3",
                "CACHE_LLM_OUTPUT_COST_PER_MILLION": "1.5",
                "CACHE_EMBEDDING_COST_PER_MILLION": "0.01",
            },
            clear=True,
        ):
            settings = PortfolioSettings.from_env(env_file="/does/not/exist")
        self.assertEqual(settings.cache_distance_threshold, 0.08)
        self.assertEqual(settings.cache_ttl_seconds, 900)
        self.assertEqual(settings.cache_llm_input_cost_per_million, 0.3)
        self.assertEqual(settings.cache_llm_output_cost_per_million, 1.5)
        self.assertEqual(settings.cache_embedding_cost_per_million, 0.01)

    def test_explicit_redis_url_takes_precedence(self) -> None:
        with patch.dict(
            os.environ,
            {
                "REDIS_URL": "rediss://example.com:6380/4",
                "REDIS_HOST": "ignored.example.com",
            },
            clear=True,
        ):
            settings = PortfolioSettings.from_env(env_file="/does/not/exist")
        self.assertEqual(settings.redis_url, "rediss://example.com:6380/4")


if __name__ == "__main__":
    unittest.main()
