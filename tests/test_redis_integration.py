from __future__ import annotations

import struct
import unittest
import uuid
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from redis.exceptions import RedisError, ResponseError
from redisvl.index import SearchIndex

from agentic.Memory.agentic_memory import (
    EMBEDDING_DIMENSIONS,
    MemoryRepository,
    create_memory_schema,
)
from RAG.User_role_based_rag import ensure_citation_schema
from redis_ai_portfolio.config import PortfolioSettings
from redis_ai_portfolio.redis import create_redis_client


class RedisIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = PortfolioSettings.from_env()
        cls.client = create_redis_client(cls.settings.redis_url)
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

    def test_json_round_trip_is_namespace_scoped(self) -> None:
        key = self.settings.redis_name("test", str(uuid.uuid4()), "json")
        try:
            self.client.json().set(key, "$", {"role": "finance", "active": True})
            self.assertEqual(
                self.client.json().get(key),
                {"role": "finance", "active": True},
            )
        finally:
            self.client.delete(key)

    def test_dropping_owned_index_preserves_unrelated_keys(self) -> None:
        run_id = str(uuid.uuid4())
        base = self.settings.redis_name("test", run_id)
        index_name = f"{base}:idx"
        document_prefix = f"{base}:document:"
        document_key = f"{document_prefix}1"
        sentinel_key = f"{base}:sentinel"
        vector = struct.pack("<2f", 1.0, 0.0)

        self.client.set(sentinel_key, "preserve")
        try:
            self.client.execute_command(
                "FT.CREATE",
                index_name,
                "ON",
                "HASH",
                "PREFIX",
                1,
                document_prefix,
                "SCHEMA",
                "title",
                "TEXT",
                "embedding",
                "VECTOR",
                "FLAT",
                6,
                "TYPE",
                "FLOAT32",
                "DIM",
                2,
                "DISTANCE_METRIC",
                "COSINE",
            )
            self.client.hset(document_key, mapping={"title": "example", "embedding": vector})
            result = self.client.execute_command(
                "FT.SEARCH",
                index_name,
                "*=>[KNN 1 @embedding $vector AS distance]",
                "PARAMS",
                2,
                "vector",
                vector,
                "DIALECT",
                2,
            )
            self.assertEqual(result[0], 1)

            self.client.execute_command("FT.DROPINDEX", index_name, "DD")
            self.assertFalse(self.client.exists(document_key))
            self.assertEqual(self.client.get(sentinel_key), b"preserve")
        finally:
            try:
                self.client.execute_command("FT.DROPINDEX", index_name, "DD")
            except ResponseError:
                pass
            keys = list(self.client.scan_iter(match=f"{base}:*", count=100))
            if keys:
                self.client.unlink(*keys)

    def test_reopening_memory_index_preserves_long_term_memory(self) -> None:
        run_id = str(uuid.uuid4())
        settings = replace(
            self.settings,
            redis_namespace=f"phase2-memory-test:{run_id}",
        )
        schema = create_memory_schema(settings)
        index = SearchIndex(schema=schema, redis_client=self.client)
        try:
            index.create(overwrite=False)
            keys = index.load(
                [
                    {
                        "memory_id": "memory-1",
                        "user_id": "alice",
                        "thread_id": "trip-1",
                        "memory_type": "episodic",
                        "content": "Prefers aisle seats",
                        "metadata": "{}",
                        "created_at": "2026-08-19T00:00:00",
                        "embedding": [0.0] * EMBEDDING_DIMENSIONS,
                    }
                ],
                id_field="memory_id",
            )

            reopened = SearchIndex(schema=schema, redis_client=self.client)
            reopened.create(overwrite=False)

            self.assertEqual(len(keys), 1)
            self.assertTrue(self.client.exists(keys[0]))
            self.assertEqual(self.client.ttl(keys[0]), -1)
            repository = MemoryRepository(index=index, vectorizer=object())
            self.assertFalse(repository.delete("memory-1", user_id="bob"))
            self.assertTrue(self.client.exists(keys[0]))
            self.assertTrue(repository.delete("memory-1", user_id="alice"))
            self.assertFalse(self.client.exists(keys[0]))
        finally:
            try:
                index.delete(drop=True)
            except RedisError:
                pass

    def test_stm_checkpoint_keys_have_a_sliding_ttl(self) -> None:
        run_id = str(uuid.uuid4())
        checkpoint_prefix = self.settings.redis_name("test", run_id, "checkpoint")
        write_prefix = self.settings.redis_name("test", run_id, "checkpoint-write")
        saver = RedisSaver(
            redis_client=self.client,
            ttl={"default_ttl": 1, "refresh_on_read": True},
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_write_prefix=write_prefix,
        )
        index_names = {checkpoint_prefix, write_prefix}
        thread_id = f"thread-{run_id}"
        try:
            saver.setup()
            graph = (
                StateGraph(MessagesState)
                .add_node(
                    "reply",
                    lambda _: {"messages": [AIMessage(content="checkpointed")]},
                )
                .add_edge(START, "reply")
                .add_edge("reply", END)
                .compile(checkpointer=saver)
            )
            graph.invoke(
                {"messages": [HumanMessage(content="hello")]},
                {"configurable": {"thread_id": thread_id}},
            )
            keys = [
                *self.client.scan_iter(match=f"{checkpoint_prefix}:*", count=100),
                *self.client.scan_iter(match=f"{write_prefix}:*", count=100),
            ]
            self.assertTrue(keys)
            self.assertTrue(all(0 < self.client.ttl(key) <= 60 for key in keys))
        finally:
            saver.delete_thread(thread_id)
            for index_name in index_names:
                try:
                    self.client.execute_command("FT.DROPINDEX", index_name, "DD")
                except ResponseError:
                    pass

    def test_existing_rag_index_gains_citation_fields_without_data_loss(self) -> None:
        run_id = str(uuid.uuid4())
        base = self.settings.redis_name("test", run_id, "rag-migration")
        index_name = f"{base}:idx"
        prefix = f"{base}:document:"
        document_key = f"{prefix}1"
        try:
            self.client.execute_command(
                "FT.CREATE",
                index_name,
                "ON",
                "JSON",
                "PREFIX",
                1,
                prefix,
                "SCHEMA",
                "$.content",
                "AS",
                "content",
                "TEXT",
            )
            self.client.json().set(
                document_key,
                "$",
                {"content": "revenue", "source": "report.pdf", "page": 7},
            )

            ensure_citation_schema(self.client, index_name)
            ensure_citation_schema(self.client, index_name)

            result = self.client.execute_command(
                "FT.SEARCH",
                index_name,
                "*",
                "RETURN",
                2,
                "source",
                "page",
                "DIALECT",
                2,
            )
            self.assertEqual(result[0], 1)
            self.assertTrue(self.client.exists(document_key))
        finally:
            try:
                self.client.execute_command("FT.DROPINDEX", index_name, "DD")
            except ResponseError:
                pass


if __name__ == "__main__":
    unittest.main()
