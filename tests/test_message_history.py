from __future__ import annotations

import io
import unittest
import uuid
from contextlib import redirect_stdout
from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from redis.exceptions import RedisError, ResponseError
from redisvl.utils.vectorize import CustomVectorizer

from llm_message_history import Multiple_sessions as message_history
from redis_ai_portfolio.config import PortfolioSettings
from redis_ai_portfolio.redis import create_redis_client


def unit_settings() -> PortfolioSettings:
    return PortfolioSettings(
        redis_url="redis://localhost:6379/0",
        redis_namespace="message-history-unit",
        openai_api_key="test-key",
        openai_model="gpt-5.6-luna",
        openai_embedding_model="text-embedding-3-small",
        cache_distance_threshold=0.2,
        cache_ttl_seconds=3600,
    )


class FakeHistory:
    def __init__(self) -> None:
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.add_calls: list[tuple[list[dict[str, str]], str]] = []
        self.recent_calls: list[str] = []

    def add_messages(self, messages, *, session_tag: str) -> None:
        copied = [dict(message) for message in messages]
        self.add_calls.append((copied, session_tag))
        self.sessions.setdefault(session_tag, []).extend(copied)

    def get_recent(self, *, session_tag: str):
        self.recent_calls.append(session_tag)
        return [dict(message) for message in self.sessions[session_tag]]

class FakeConversationClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def converse(self, prompt: str, context: list[dict[str, str]]) -> str:
        self.calls.append((prompt, context))
        return f"response-{len(self.calls)}"


class MessageHistoryUnitTests(unittest.TestCase):
    def test_remap_converts_supported_roles_without_mutating_context(self) -> None:
        context = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "current answer"},
            {"role": "llm", "content": "legacy answer"},
        ]
        original = [dict(statement) for statement in context]
        client = object.__new__(message_history.OpenAIClient)

        remapped = client.remap(context)

        self.assertEqual(
            remapped,
            [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "current answer"},
                {"role": "assistant", "content": "legacy answer"},
            ],
        )
        self.assertEqual(context, original)

    def test_remap_rejects_unknown_roles(self) -> None:
        client = object.__new__(message_history.OpenAIClient)

        with self.assertRaisesRegex(ValueError, "Unknown chat role"):
            client.remap([{"role": "developer", "content": "hidden"}])

    def test_converse_sends_mapped_history_then_new_prompt(self) -> None:
        api = MagicMock()
        api.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="model answer"))]
        )
        with patch.object(message_history, "OpenAI", return_value=api):
            client = message_history.OpenAIClient(
                api_key="test-key",
                model="gpt-5.6-luna",
            )
            answer = client.converse(
                "new question",
                [{"role": "llm", "content": "previous answer"}],
            )

        self.assertEqual(answer, "model answer")
        api.chat.completions.create.assert_called_once_with(
            model="gpt-5.6-luna",
            messages=[
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "new question"},
            ],
        )

    def test_converse_rejects_an_empty_model_response(self) -> None:
        api = MagicMock()
        api.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        with patch.object(message_history, "OpenAI", return_value=api):
            client = message_history.OpenAIClient(api_key="test-key")
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                client.converse("question", [])

    def test_factory_applies_namespaced_index_prefix_client_and_vectorizer(self) -> None:
        settings = unit_settings()
        redis_client = MagicMock()
        vectorizer = MagicMock()
        created = MagicMock()
        with patch.object(
            message_history,
            "SemanticMessageHistory",
            return_value=created,
        ) as history_class:
            result = message_history.create_message_history(
                redis_client,
                settings=settings,
                vectorizer=vectorizer,
            )

        self.assertIs(result, created)
        history_class.assert_called_once_with(
            name="message-history-unit:idx:message-history:budgeting",
            prefix="message-history-unit:message-history:budgeting:",
            redis_client=redis_client,
            vectorizer=vectorizer,
        )

    def test_run_demo_keeps_each_persona_in_its_own_session(self) -> None:
        history = FakeHistory()
        client = FakeConversationClient()

        with redirect_stdout(io.StringIO()):
            message_history.run_demo(history, client)

        self.assertEqual(set(history.sessions), {"student", "young professional", "retired pensioner"})
        self.assertEqual(history.recent_calls, [
            "student",
            "young professional",
            "retired pensioner",
            "student",
        ])
        self.assertEqual(len(client.calls), 3)
        self.assertIn("college student", client.calls[0][1][1]["content"])
        self.assertIn("young professional", client.calls[1][1][1]["content"])
        self.assertIn("retired pensioner", client.calls[2][1][1]["content"])
        exchange_calls = history.add_calls[3:]
        self.assertEqual(
            [session_tag for _, session_tag in exchange_calls],
            ["student", "young professional", "retired pensioner"],
        )
        self.assertTrue(
            all(
                [message["role"] for message in messages] == ["user", "assistant"]
                for messages, _ in exchange_calls
            )
        )
        self.assertEqual(
            {messages[0]["content"] for messages, _ in exchange_calls},
            {"What is the single most important thing I should focus on financially?"},
        )

    def test_main_clears_then_deletes_owned_history_and_closes_clients(self) -> None:
        settings = unit_settings()
        redis_client = MagicMock()
        openai_client = MagicMock()
        history = MagicMock()
        with (
            patch.object(message_history, "SETTINGS", settings),
            patch.object(message_history, "REDIS_URL", settings.redis_url),
            patch.object(message_history, "create_redis_client", return_value=redis_client),
            patch.object(message_history, "OpenAIClient", return_value=openai_client),
            patch.object(message_history, "create_message_history", return_value=history),
            patch.object(message_history, "run_demo") as run_demo,
        ):
            message_history.main()

        redis_client.ping.assert_called_once_with()
        history.clear.assert_called_once_with()
        run_demo.assert_called_once_with(history, openai_client)
        history.delete.assert_called_once_with()
        openai_client.close.assert_called_once_with()
        redis_client.close.assert_called_once_with()

    def test_main_closes_every_initialized_resource_when_demo_fails(self) -> None:
        settings = unit_settings()
        redis_client = MagicMock()
        openai_client = MagicMock()
        history = MagicMock()
        with (
            patch.object(message_history, "SETTINGS", settings),
            patch.object(message_history, "REDIS_URL", settings.redis_url),
            patch.object(message_history, "create_redis_client", return_value=redis_client),
            patch.object(message_history, "OpenAIClient", return_value=openai_client),
            patch.object(message_history, "create_message_history", return_value=history),
            patch.object(message_history, "run_demo", side_effect=RuntimeError("demo failed")),
            self.assertRaisesRegex(RuntimeError, "demo failed"),
        ):
            message_history.main()

        history.delete.assert_called_once_with()
        openai_client.close.assert_called_once_with()
        redis_client.close.assert_called_once_with()

    def test_main_closes_redis_when_openai_client_initialization_fails(self) -> None:
        settings = unit_settings()
        redis_client = MagicMock()
        with (
            patch.object(message_history, "SETTINGS", settings),
            patch.object(message_history, "REDIS_URL", settings.redis_url),
            patch.object(message_history, "create_redis_client", return_value=redis_client),
            patch.object(
                message_history,
                "OpenAIClient",
                side_effect=RuntimeError("client initialization failed"),
            ),
            patch.object(message_history, "create_message_history") as history_factory,
            self.assertRaisesRegex(RuntimeError, "client initialization failed"),
        ):
            message_history.main()

        history_factory.assert_not_called()
        redis_client.close.assert_called_once_with()


class MessageHistoryRedisIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_settings = PortfolioSettings.from_env()
        cls.client = create_redis_client(cls.base_settings.redis_url)
        try:
            cls.client.ping()
            if not cls.client.execute_command("COMMAND", "INFO", "FT.CREATE"):
                raise unittest.SkipTest("Redis Search is unavailable")
        except RedisError as exc:
            cls.client.close()
            raise unittest.SkipTest(f"Redis integration unavailable: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def setUp(self) -> None:
        self.settings = replace(
            self.base_settings,
            redis_namespace=f"message-history-integration:{uuid.uuid4().hex}",
        )
        self.index_name = self.settings.redis_name(
            "idx", "message-history", "budgeting"
        )
        self.key_prefix = f"{self.settings.redis_name('message-history', 'budgeting')}:"
        self.vectorizer = CustomVectorizer(
            embed=lambda _text: [1.0, 0.0, 0.0, 0.0],
            dtype="float32",
        )
        self.history = message_history.create_message_history(
            self.client,
            settings=self.settings,
            vectorizer=self.vectorizer,
        )

    def tearDown(self) -> None:
        try:
            self.client.execute_command("FT.DROPINDEX", self.index_name, "DD")
        except ResponseError:
            pass
        keys = list(
            self.client.scan_iter(match=f"{self.settings.redis_namespace}:*", count=100)
        )
        if keys:
            self.client.unlink(*keys)

    def test_session_filtering_order_and_serialization_round_trip(self) -> None:
        timestamps = count(start=1)
        with patch(
            "redisvl.extensions.message_history.schema.current_timestamp",
            side_effect=lambda: float(next(timestamps)),
        ):
            self.history.add_messages(
                [
                    {"role": "system", "content": "student policy"},
                    {"role": "user", "content": "student budget"},
                ],
                session_tag="student",
            )
            self.history.add_messages(
                [
                    {"role": "system", "content": "retiree policy"},
                    {"role": "user", "content": "retiree budget"},
                ],
                session_tag="retired",
            )
            message_history.store_exchange(
                self.history,
                "student follow-up",
                "student answer",
                session_tag="student",
            )

        student = self.history.get_recent(top_k=10, session_tag="student")
        retired = self.history.get_recent(top_k=10, session_tag="retired")

        self.assertEqual(
            [message["content"] for message in student],
            [
                "student policy",
                "student budget",
                "student follow-up",
                "student answer",
            ],
        )
        self.assertEqual(
            [message["role"] for message in student],
            ["system", "user", "user", "assistant"],
        )
        self.assertEqual(
            [message["content"] for message in retired],
            ["retiree policy", "retiree budget"],
        )
        self.assertFalse(
            {message["content"] for message in student}
            & {message["content"] for message in retired}
        )
        self.assertEqual(self.history.count(session_tag="student"), 4)
        self.assertEqual(self.history.count(session_tag="retired"), 2)

    def test_clear_and_delete_are_scoped_to_owned_history(self) -> None:
        sentinel = self.settings.redis_name("sentinel")
        self.client.set(sentinel, "preserve")
        self.history.add_message(
            {"role": "user", "content": "temporary message"},
            session_tag="student",
        )

        self.history.clear()

        self.assertEqual(self.history.count(session_tag="student"), 0)
        self.assertEqual(self.client.get(sentinel), b"preserve")
        self.client.execute_command("FT.INFO", self.index_name)

        self.history.add_message(
            {"role": "user", "content": "delete with index"},
            session_tag="student",
        )
        self.history.delete()

        self.assertEqual(self.client.get(sentinel), b"preserve")
        self.assertFalse(list(self.client.scan_iter(match=f"{self.key_prefix}*", count=100)))
        with self.assertRaises(ResponseError):
            self.client.execute_command("FT.INFO", self.index_name)


if __name__ == "__main__":
    unittest.main()
