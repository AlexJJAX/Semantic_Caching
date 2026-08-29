from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from redis_ai_portfolio.config import PortfolioSettings

ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, relative_path: str):
    """Load a portfolio script without executing its __main__ entry point."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FLEX = load_script_module(
    "portfolio_phase2_flex",
    "agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py",
)
MEMORY = load_script_module(
    "portfolio_phase2_memory",
    "agentic/Memory/agentic_memory.py",
)


def settings() -> PortfolioSettings:
    return PortfolioSettings(
        redis_url="redis://localhost:6379/0",
        redis_namespace="phase2-test",
        openai_api_key=None,
        openai_model="gpt-5.6-luna",
        openai_embedding_model="text-embedding-3-small",
        cache_distance_threshold=0.1,
        cache_ttl_seconds=3600,
    )


class StubChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "phase2-stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="stub answer"))])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class FakeVectorizer:
    def embed(self, _: str) -> list[float]:
        return [0.0] * MEMORY.EMBEDDING_DIMENSIONS


class FakeIndex:
    def __init__(self, query_results=None):
        self.query_results = list(query_results or [])
        self.loaded: list[tuple[list[dict], str | None]] = []
        self.dropped: list[str] = []

    def query(self, _):
        return self.query_results

    def load(self, data, id_field=None):
        records = list(data)
        self.loaded.append((records, id_field))
        return [f"phase2-test:agent-memory:{records[0]['memory_id']}"]

    def drop_keys(self, key):
        self.dropped.append(key)
        return 1


class FlexRoutingTests(unittest.TestCase):
    def test_relevant_documents_generate_immediately(self) -> None:
        state = {"messages": [], "documents_relevant": True, "rewrite_count": 0}
        self.assertEqual(FLEX.route_after_grading(state, max_rewrites=2), "generate")

    def test_irrelevant_documents_stop_at_rewrite_limit(self) -> None:
        state = {"messages": [], "documents_relevant": False, "rewrite_count": 2}
        self.assertEqual(FLEX.route_after_grading(state, max_rewrites=2), "not_found")

    def test_irrelevant_documents_rewrite_below_limit(self) -> None:
        state = {"messages": [], "documents_relevant": False, "rewrite_count": 1}
        self.assertEqual(FLEX.route_after_grading(state, max_rewrites=2), "rewrite")


class MemoryRepositoryTests(unittest.TestCase):
    def test_schema_indexes_thread_id_and_matches_embedding_dimensions(self) -> None:
        schema = MEMORY.create_memory_schema(settings()).to_dict()
        fields = {field["name"]: field for field in schema["fields"]}
        self.assertEqual(fields["thread_id"]["type"], "tag")
        self.assertEqual(
            fields["embedding"]["attrs"]["dims"],
            MEMORY.EMBEDDING_DIMENSIONS,
        )

    def test_store_uses_json_metadata_and_stable_id_field(self) -> None:
        index = FakeIndex()
        repository = MEMORY.MemoryRepository(index=index, vectorizer=FakeVectorizer())

        created = repository.store(
            "Prefers aisle seats",
            MEMORY.MemoryType.EPISODIC,
            user_id="alice",
            thread_id="trip-1",
            metadata={"source": "chat"},
        )

        self.assertTrue(created)
        records, id_field = index.loaded[0]
        self.assertEqual(id_field, "memory_id")
        self.assertEqual(records[0]["memory_type"], "episodic")
        self.assertEqual(records[0]["thread_id"], "trip-1")
        metadata = json.loads(records[0]["metadata"])
        self.assertEqual(metadata["source"], "chat")
        self.assertEqual(metadata["provenance"]["source"], "manual")
        self.assertEqual(metadata["provenance"]["stored_by"], "application")
        self.assertEqual(metadata["provenance"]["thread_id"], "trip-1")

    def test_delete_is_scoped_to_the_requesting_user(self) -> None:
        index = FakeIndex(
            query_results=[
                {
                    "id": "phase2-test:agent-memory:memory-1",
                    "memory_id": "memory-1",
                }
            ]
        )
        repository = MEMORY.MemoryRepository(index=index, vectorizer=FakeVectorizer())

        self.assertTrue(repository.delete("memory-1", user_id="alice"))
        self.assertEqual(index.dropped, ["phase2-test:agent-memory:memory-1"])

    def test_missing_memory_is_not_deleted(self) -> None:
        index = FakeIndex()
        repository = MEMORY.MemoryRepository(index=index, vectorizer=FakeVectorizer())

        self.assertFalse(repository.delete("memory-1", user_id="alice"))
        self.assertEqual(index.dropped, [])

    def test_duplicate_memory_is_not_loaded(self) -> None:
        index = FakeIndex(query_results=[{"memory_id": "existing"}])
        repository = MEMORY.MemoryRepository(index=index, vectorizer=FakeVectorizer())

        created = repository.store(
            "Prefers aisle seats",
            MEMORY.MemoryType.EPISODIC,
            user_id="alice",
        )

        self.assertFalse(created)
        self.assertEqual(index.loaded, [])

    def test_tool_runtime_config_is_injected_not_exposed_to_model(self) -> None:
        repository = MEMORY.MemoryRepository(index=FakeIndex(), vectorizer=FakeVectorizer())
        tools = MEMORY.create_memory_tools(repository)
        for memory_tool in tools:
            properties = memory_tool.args_schema.model_json_schema()["properties"]
            self.assertNotIn("config", properties)

    def test_retrieval_tool_exposes_memory_id_for_explicit_deletion(self) -> None:
        index = FakeIndex(
            query_results=[
                {
                    "id": "phase2-test:agent-memory:memory-1",
                    "memory_id": "memory-1",
                    "user_id": "alice",
                    "thread_id": "trip-1",
                    "memory_type": "episodic",
                    "content": "Prefers aisle seats",
                    "created_at": "2026-08-20T10:00:00+00:00",
                    "metadata": "{}",
                }
            ]
        )
        repository = MEMORY.MemoryRepository(index=index, vectorizer=FakeVectorizer())
        retrieve_tool = MEMORY.create_memory_tools(repository)[1]

        result = retrieve_tool.invoke(
            {"query": "seat preference"},
            config={"configurable": {"user_id": "alice", "thread_id": "trip-1"}},
        )

        self.assertIn("ID memory-1", result)

    def test_connect_preserves_index_and_configures_exact_embedding_dimensions(self) -> None:
        configured = replace(settings(), openai_api_key="test-key")
        with (
            patch.object(MEMORY, "SearchIndex") as search_index,
            patch.object(MEMORY, "OpenAIEmbeddings") as embeddings,
        ):
            repository = MEMORY.MemoryRepository.connect(
                configured,
                redis_client=object(),
            )

        search_index.return_value.create.assert_called_once_with(overwrite=False)
        embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            api_key="test-key",
            dimensions=MEMORY.EMBEDDING_DIMENSIONS,
            max_retries=2,
            request_timeout=10,
        )
        self.assertIs(repository.vectorizer, embeddings.return_value)

    def test_stm_ttl_is_sliding_and_does_not_apply_to_ltm(self) -> None:
        configured = replace(
            settings(),
            stm_ttl_minutes=45,
            stm_refresh_ttl_on_read=True,
        )
        self.assertEqual(
            MEMORY.checkpoint_ttl_config(configured),
            {"default_ttl": 45, "refresh_on_read": True},
        )


class TravelGraphTests(unittest.TestCase):
    def test_checkpointed_turns_do_not_require_resubmitting_history(self) -> None:
        repository = MEMORY.MemoryRepository(index=FakeIndex(), vectorizer=FakeVectorizer())
        graph = MEMORY.build_travel_graph(
            repository,
            model=StubChatModel(),
            summarizer=StubChatModel(),
            checkpointer=InMemorySaver(),
            summarization_threshold=100,
        )
        config = {"configurable": {"thread_id": "thread-1", "user_id": "alice"}}

        first = graph.invoke({"messages": [HumanMessage(content="Hello")]}, config)
        second = graph.invoke({"messages": [HumanMessage(content="Plan Paris")]}, config)

        self.assertEqual(len(first["messages"]), 2)
        self.assertEqual(len(second["messages"]), 4)
        self.assertEqual(second["messages"][0].content, "Hello")
        self.assertEqual(second["messages"][2].content, "Plan Paris")


if __name__ == "__main__":
    unittest.main()
