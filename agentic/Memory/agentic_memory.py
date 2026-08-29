"""Import-safe Redis short- and long-term memory travel assistant."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import ulid
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorRangeQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema

from redis_ai_portfolio.config import PortfolioSettings, get_settings, redact_redis_url
from redis_ai_portfolio.redis import create_redis_client

logger = logging.getLogger(__name__)

SYSTEM_USER_ID = "system"
EMBEDDING_DIMENSIONS = 512
MEMORY_DEDUP_DISTANCE = 0.1
MEMORY_RETRIEVAL_DISTANCE = 0.3
MESSAGE_SUMMARIZATION_THRESHOLD = 6
GRAPH_RECURSION_LIMIT = 12

TRAVEL_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a travel assistant helping users plan trips. Use short-term conversation "
        "context and the memory tools when useful. Store durable user preferences and past "
        "experiences as episodic memories; store reusable travel facts as semantic memories. "
        "Be helpful, personal, concise, and explicit when information is uncertain."
    )
)


class MemoryType(StrEnum):
    """Long-term memory categories."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class Memory(BaseModel):
    """A candidate long-term memory."""

    content: str
    memory_type: MemoryType
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memories(BaseModel):
    """Structured collection of candidate memories."""

    memories: list[Memory]


class StoredMemory(Memory):
    """A long-term memory returned from Redis."""

    id: str
    memory_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = SYSTEM_USER_ID
    thread_id: str | None = None


class MemoryProvenance(BaseModel):
    """Traceability attached to every newly stored long-term memory."""

    source: str
    source_id: str | None = None
    stored_by: str
    thread_id: str | None = None
    created_at: datetime


class TravelState(MessagesState):
    """Message state persisted by the Redis checkpointer."""


def create_memory_schema(
    settings: PortfolioSettings,
    *,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> IndexSchema:
    """Create the persistent long-term-memory schema."""
    return IndexSchema.from_dict(
        {
            "index": {
                "name": settings.redis_name("idx", "agent-memory"),
                "prefix": settings.redis_name("agent-memory"),
                "key_separator": ":",
                "storage_type": "json",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "memory_type", "type": "tag"},
                {"name": "metadata", "type": "text"},
                {"name": "created_at", "type": "text"},
                {"name": "user_id", "type": "tag"},
                {"name": "thread_id", "type": "tag"},
                {"name": "memory_id", "type": "tag"},
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


@dataclass(slots=True)
class MemoryRepository:
    """Long-term memory operations with explicit index/vectorizer dependencies."""

    index: SearchIndex
    vectorizer: Any

    @classmethod
    def connect(
        cls,
        settings: PortfolioSettings,
        redis_client: Redis,
    ) -> MemoryRepository:
        """Open the memory index without overwriting existing records."""
        schema = create_memory_schema(settings)
        index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
            validate_on_load=True,
        )
        index.create(overwrite=False)
        vectorizer = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            dimensions=EMBEDDING_DIMENSIONS,
            max_retries=2,
            request_timeout=10,
        )
        return cls(index=index, vectorizer=vectorizer)

    def embed(self, content: str) -> list[float]:
        """Embed text through a LangChain or RedisVL-compatible dependency."""
        if hasattr(self.vectorizer, "embed_query"):
            return self.vectorizer.embed_query(content)
        return self.vectorizer.embed(content)

    def similar_memory_exists(
        self,
        content: str,
        memory_type: MemoryType,
        *,
        user_id: str = SYSTEM_USER_ID,
        thread_id: str | None = None,
        distance_threshold: float = MEMORY_DEDUP_DISTANCE,
    ) -> bool:
        """Return whether a user already has a semantically similar memory."""
        filters = (Tag("user_id") == user_id) & (
            Tag("memory_type") == memory_type.value
        )
        if thread_id:
            filters = filters & (Tag("thread_id") == thread_id)

        query = VectorRangeQuery(
            vector=self.embed(content),
            num_results=1,
            vector_field_name="embedding",
            filter_expression=filters,
            distance_threshold=distance_threshold,
            return_fields=["memory_id"],
        )
        return bool(self.index.query(query))

    def store(
        self,
        content: str,
        memory_type: MemoryType,
        *,
        user_id: str = SYSTEM_USER_ID,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "manual",
        source_id: str | None = None,
        stored_by: str = "application",
    ) -> bool:
        """Store a memory unless the same user already has a close semantic match."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty")

        if self.similar_memory_exists(
            clean_content,
            memory_type,
            user_id=user_id,
            thread_id=thread_id,
        ):
            return False

        memory_id = str(ulid.ULID())
        created_at = datetime.now(timezone.utc)
        provenance = MemoryProvenance(
            source=source,
            source_id=source_id,
            stored_by=stored_by,
            thread_id=thread_id,
            created_at=created_at,
        )
        stored_metadata = dict(metadata or {})
        stored_metadata["provenance"] = provenance.model_dump(mode="json")
        self.index.load(
            [
                {
                    "user_id": user_id,
                    "content": clean_content,
                    "memory_type": memory_type.value,
                    "metadata": json.dumps(stored_metadata, sort_keys=True),
                    "created_at": created_at.isoformat(),
                    "embedding": self.embed(clean_content),
                    "memory_id": memory_id,
                    "thread_id": thread_id or "",
                }
            ],
            id_field="memory_id",
        )
        return True

    def delete(self, memory_id: str, *, user_id: str) -> bool:
        """Delete one long-term memory only when it belongs to the requesting user."""
        clean_memory_id = memory_id.strip()
        if not clean_memory_id:
            raise ValueError("memory_id cannot be empty")

        results = self.index.query(
            FilterQuery(
                filter_expression=(Tag("memory_id") == clean_memory_id)
                & (Tag("user_id") == user_id),
                return_fields=["memory_id"],
                num_results=1,
            )
        )
        if not results:
            return False
        return self.index.drop_keys(results[0]["id"]) == 1

    def retrieve(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        user_id: str = SYSTEM_USER_ID,
        thread_id: str | None = None,
        distance_threshold: float = MEMORY_RETRIEVAL_DISTANCE,
        limit: int = 5,
    ) -> list[StoredMemory]:
        """Retrieve user-scoped memories with optional type and thread filters."""
        if limit < 1:
            raise ValueError("limit must be at least 1")

        filters = Tag("user_id") == user_id
        if memory_types:
            type_filter = Tag("memory_type") == memory_types[0].value
            for memory_type in memory_types[1:]:
                type_filter = type_filter | (Tag("memory_type") == memory_type.value)
            filters = filters & type_filter
        if thread_id:
            filters = filters & (Tag("thread_id") == thread_id)

        results = self.index.query(
            VectorRangeQuery(
                vector=self.embed(query),
                return_fields=[
                    "content",
                    "memory_type",
                    "metadata",
                    "created_at",
                    "memory_id",
                    "thread_id",
                    "user_id",
                ],
                num_results=limit,
                vector_field_name="embedding",
                dialect=2,
                distance_threshold=distance_threshold,
                filter_expression=filters,
            )
        )

        memories: list[StoredMemory] = []
        for document in results:
            raw_metadata = document.get("metadata") or "{}"
            metadata = (
                json.loads(raw_metadata)
                if isinstance(raw_metadata, str)
                else dict(raw_metadata)
            )
            memories.append(
                StoredMemory(
                    id=document["id"],
                    memory_id=document["memory_id"],
                    user_id=document["user_id"],
                    thread_id=document.get("thread_id") or None,
                    memory_type=MemoryType(document["memory_type"]),
                    content=document["content"],
                    created_at=document["created_at"],
                    metadata=metadata,
                )
            )
        return memories


def create_memory_tools(repository: MemoryRepository) -> list[BaseTool]:
    """Create agent tools bound to one long-term-memory repository."""

    @tool
    def store_memory(
        content: str,
        memory_type: MemoryType,
        config: RunnableConfig,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a durable episodic preference/experience or semantic travel fact."""
        configurable = config.get("configurable", {})
        run_id = config.get("run_id")
        created = repository.store(
            content,
            memory_type,
            user_id=configurable.get("user_id", SYSTEM_USER_ID),
            thread_id=configurable.get("thread_id"),
            metadata=metadata,
            source="agent-tool",
            source_id=str(run_id) if run_id else configurable.get("thread_id"),
            stored_by=configurable.get("user_id", SYSTEM_USER_ID),
        )
        if created:
            return f"Stored {memory_type.value} memory: {content}"
        return "Skipped storage because a similar memory already exists."

    @tool
    def retrieve_memories(
        query: str,
        config: RunnableConfig,
        memory_types: list[MemoryType] | None = None,
        limit: int = 5,
    ) -> str:
        """Retrieve durable memories relevant to a travel-planning query."""
        configurable = config.get("configurable", {})
        memories = repository.retrieve(
            query,
            memory_types=memory_types,
            user_id=configurable.get("user_id", SYSTEM_USER_ID),
            limit=limit,
        )
        if not memories:
            return "No relevant memories found."
        return "Long-term memories:\n" + "\n".join(
            f"- ID {memory.memory_id} [{memory.memory_type.value}] {memory.content}"
            for memory in memories
        )

    @tool
    def delete_memory(memory_id: str, config: RunnableConfig) -> str:
        """Delete a memory by ID only when the user explicitly requests deletion."""
        configurable = config.get("configurable", {})
        deleted = repository.delete(
            memory_id,
            user_id=configurable.get("user_id", SYSTEM_USER_ID),
        )
        if deleted:
            return f"Deleted memory {memory_id}."
        return "No matching memory was found for this user."

    return [store_memory, retrieve_memories, delete_memory]


def checkpoint_ttl_config(settings: PortfolioSettings) -> dict[str, int | bool]:
    """Return the sliding expiry policy applied only to STM checkpoint keys."""
    return {
        "default_ttl": settings.stm_ttl_minutes,
        "refresh_on_read": settings.stm_refresh_ttl_on_read,
    }


def _message_text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "User"
    if isinstance(message, AIMessage):
        return "Assistant"
    return message.type.title()


def build_travel_graph(
    repository: MemoryRepository,
    *,
    model: Any,
    summarizer: Any,
    checkpointer: RedisSaver,
    summarization_threshold: int = MESSAGE_SUMMARIZATION_THRESHOLD,
) -> Any:
    """Build one checkpointed graph; tool execution is handled by ToolNode."""
    if summarization_threshold < 2:
        raise ValueError("summarization_threshold must be at least 2")

    tools = create_memory_tools(repository)
    model_with_tools = model.bind_tools(tools)
    retry_policy = RetryPolicy(max_attempts=2, initial_interval=0.5)

    def call_agent(state: TravelState) -> dict:
        response = model_with_tools.invoke([TRAVEL_SYSTEM_PROMPT, *state["messages"]])
        return {"messages": [response]}

    def summarize_conversation(state: TravelState) -> dict:
        messages = state["messages"]
        if len(messages) < summarization_threshold:
            return {}

        transcript = "\n".join(
            f"{_message_role(message)}: {_message_text(message)}" for message in messages
        )
        response = summarizer.invoke(
            [
                SystemMessage(
                    content=(
                        "Summarize this travel conversation. Preserve user preferences, trip "
                        "details, decisions, and unresolved questions in one concise paragraph."
                    )
                ),
                HumanMessage(content=transcript),
            ]
        )
        summary = SystemMessage(content=f"Conversation summary: {_message_text(response)}")
        removable = [message for message in messages if message.id is not None]
        if not removable:
            return {"messages": [summary]}
        latest = messages[-1]
        removals = [RemoveMessage(id=message.id) for message in removable]
        return {"messages": [*removals, summary, latest]}

    return (
        StateGraph(TravelState)
        .add_node("agent", call_agent, retry_policy=retry_policy)
        .add_node("tools", ToolNode(tools, handle_tool_errors=True))
        .add_node("summarize", summarize_conversation, retry_policy=retry_policy)
        .add_edge(START, "agent")
        .add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: "summarize"},
        )
        .add_edge("tools", "agent")
        .add_edge("summarize", END)
        .compile(checkpointer=checkpointer)
    )


@dataclass(slots=True)
class TravelMemoryApplication:
    """Owned runtime resources for the memory-enabled travel graph."""

    graph: Any
    repository: MemoryRepository
    redis_client: Redis

    def close(self) -> None:
        self.redis_client.close()


def create_travel_memory_application(
    settings: PortfolioSettings | None = None,
) -> TravelMemoryApplication:
    """Initialize Redis, persistent indexes, models, and the travel graph."""
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

    redis_client = create_redis_client(settings.redis_url)
    try:
        redis_client.ping()
        repository = MemoryRepository.connect(settings, redis_client)
        saver = RedisSaver(
            redis_client=redis_client,
            ttl=checkpoint_ttl_config(settings),
            checkpoint_prefix=settings.redis_name("checkpoint", "travel-agent"),
            checkpoint_write_prefix=settings.redis_name(
                "checkpoint-write", "travel-agent"
            ),
        )
        saver.setup()
        graph = build_travel_graph(
            repository,
            model=ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.7,
            ),
            summarizer=ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.3,
            ),
            checkpointer=saver,
        )
        return TravelMemoryApplication(graph, repository, redis_client)
    except Exception:
        redis_client.close()
        raise


def main(thread_id: str = "demo-thread", user_id: str = "demo-user") -> None:
    """Run an interactive conversation without duplicating checkpointed state."""
    settings = get_settings()
    print(f"Connecting to Redis at {redact_redis_url(settings.redis_url)}")
    app = create_travel_memory_application(settings)
    config = RunnableConfig(
        configurable={"thread_id": thread_id, "user_id": user_id},
        recursion_limit=GRAPH_RECURSION_LIMIT,
    )

    print("Welcome to the Travel Assistant! Type 'exit' or 'quit' to stop.")
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Thank you for using the Travel Assistant. Goodbye!")
                break

            final_state: dict[str, Any] | None = None
            for state in app.graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values",
            ):
                final_state = state

            ai_messages = [
                message
                for message in (final_state or {}).get("messages", [])
                if isinstance(message, AIMessage)
            ]
            answer = (
                _message_text(ai_messages[-1])
                if ai_messages
                else "I'm sorry, I couldn't produce a response."
            )
            print(f"\nAssistant: {answer}")
    finally:
        app.close()


if __name__ == "__main__":
    selected_user = input("Enter a user ID (default: demo-user): ").strip() or "demo-user"
    selected_thread = (
        input("Enter a thread ID (default: demo-thread): ").strip() or "demo-thread"
    )
    main(thread_id=selected_thread, user_id=selected_user)
