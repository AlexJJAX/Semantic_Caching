# Redis STM/LTM Travel Assistant with LangGraph & GPT-5.6 Luna

![Tests](https://img.shields.io/badge/focused%20tests-10%20passing-2E7D32)
![Integration](https://img.shields.io/badge/Redis%20integration-2%20passing-00796B)
![Model](https://img.shields.io/badge/model-gpt--5.6--luna-412991)
![Redis](https://img.shields.io/badge/Redis-8%20Search%20%7C%20JSON-DC382D)
![License](https://img.shields.io/badge/license-MIT-455A64)

A conversational travel assistant that demonstrates two deliberately different forms of
memory: expiring conversation state for continuity and durable, searchable facts for
personalization.

LangGraph coordinates one checkpointed agent loop. `RedisSaver` stores short-term memory (STM),
while RedisVL and Redis Search store long-term memory (LTM) as provenance-bearing vectors.
GPT-5.6 Luna decides when to store, retrieve, or explicitly delete durable memories.

This is a generic, demonstrational working primitive intended to showcase Redis short-term memory
(STM) and long-term memory (LTM) with deliberately different lifecycle rules. It is not intended
or suitable for production use: the example identities, model-selected memories, TTL, provenance,
and deletion tools illustrate mechanisms rather than a complete personal-data or memory-governance
system.

## Architecture Overview

| Component                            | Responsibility                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| `TravelState`                        | Holds LangGraph messages persisted by the Redis checkpointer                               |
| `RedisSaver`                         | Restores STM by `thread_id` and applies the configured sliding expiry policy               |
| `MemoryRepository`                   | Owns typed LTM storage, semantic deduplication, retrieval, and user-scoped deletion        |
| `store_memory`                       | Stores a novel episodic or semantic memory with provenance                                 |
| `retrieve_memories`                  | Returns up to five relevant memories, including stable IDs needed for deletion             |
| `delete_memory`                      | Deletes one LTM record only when its `memory_id` belongs to the current user               |
| Agent node                           | Uses conversation state and chooses whether to call a memory tool                          |
| Summarizer node                      | Compacts long message histories into a system summary while preserving the latest response |
| `create_travel_memory_application()` | Initializes the shared Redis client, LTM index, checkpointer, models, and graph            |
| Interactive CLI                      | Sends only the new turn, streams the restored graph state, and closes resources on exit    |

All executable behavior lives in [`agentic_memory.py`](./agentic_memory.py). Importing the
module does not connect to Redis, create indexes, or call OpenAI.

## Why two memory layers

STM and LTM solve different problems and therefore have different retention rules:

| Property          | Short-term memory                       | Long-term memory                                           |
| ----------------- | --------------------------------------- | ---------------------------------------------------------- |
| Purpose           | Preserve conversational continuity      | Reuse preferences, experiences, and travel facts           |
| Redis integration | LangGraph `RedisSaver`                  | RedisVL `SearchIndex` over RedisJSON                       |
| Primary scope     | `thread_id`                             | `user_id`; `thread_id` is also stored and indexed          |
| Content           | Exact messages plus compacted summaries | Selected episodic and semantic memories                    |
| Write path        | Automatic graph checkpointing           | Explicit `store_memory` tool call chosen by the agent      |
| Read path         | Automatically restored before each turn | Semantic `retrieve_memories` tool call chosen by the agent |
| Default retention | 1,440-minute sliding TTL                | No TTL                                                     |
| Deletion          | Expires by policy                       | Explicit, user-scoped deletion by `memory_id`              |

The terminal sends only the newest `HumanMessage`. LangGraph restores prior state for the
selected thread, so the caller never resubmits the full checkpointed transcript.

## What it demonstrates

- One `StateGraph` and one Redis checkpointer, without a nested agent graph.
- Sliding expiration for temporary checkpoint data.
- Durable LTM that remains available across threads and application restarts.
- A RedisJSON schema with exact-match tags and a 512-dimensional cosine vector.
- Separate distance thresholds for duplicate prevention and memory retrieval.
- User-scoped retrieval and deletion enforced in the Redis query layer.
- Episodic and semantic memory categories with stable ULID identifiers.
- Provenance containing source, source ID, storing actor, thread, and timestamp.
- Runtime identity injection that is hidden from the model's tool schemas.
- Reducer-safe conversation compaction with LangGraph `RemoveMessage` updates.
- Bounded OpenAI retries, Redis timeouts, and graph recursion.
- Import-safe factories and deterministic resource cleanup.

## Key Design Decisions

- **Separate retention from importance** — checkpoint data is temporary and expires; LTM is
  intentionally durable and requires an explicit deletion request.
- **Scope access before vector ranking** — retrieval applies the `user_id` tag filter inside the
  Redis vector query. Another user's memories are not retrieved and filtered afterward.
- **Authorize deletion in the query** — deletion requires both `memory_id` and `user_id` to
  match before the underlying Redis key can be removed.
- **Keep identity out of model arguments** — LangGraph injects `user_id`, `thread_id`, and run
  metadata through `RunnableConfig`; those fields are absent from the tool schemas visible to
  the model.
- **Attach provenance at write time** — every stored record includes an application-controlled
  provenance envelope alongside optional caller metadata.
- **Deduplicate before persistence** — a cosine-distance range query checks for a similar memory
  of the same type, user, and current thread before creating a new vector.
- **Make deletion discoverable** — retrieval includes the stable memory ID in tool output, so a
  later explicit request can identify the record to remove.
- **Compact checkpoint state, not LTM** — after six messages, the summarizer replaces older
  messages with a concise system summary. Durable RedisVL memories are unaffected.
- **Preserve existing data** — startup uses `create(overwrite=False)` for the LTM index and
  `setup()` for checkpoint indexes, allowing state to survive restarts.

## Identity and isolation

The example uses `user_id` and `thread_id` for different boundaries:

| Scenario                         | STM behavior                         | LTM behavior                                               |
| -------------------------------- | ------------------------------------ | ---------------------------------------------------------- |
| Same user, same thread           | Resumes unexpired conversation state | Searches that user's durable memories                      |
| Same user, new thread            | Starts separate conversation state   | Can still retrieve that user's memories from other threads |
| Different user, unique thread    | Separate conversation state          | Separate, user-filtered durable memories                   |
| Different user, reused thread ID | Checkpoint state can collide         | LTM remains filtered by `user_id`                          |

For a production application, derive a globally unique thread identifier from authenticated
tenant, user, and conversation IDs. The interactive identifiers in this demonstration are
caller-supplied labels, not authentication credentials.

## Long-term memory schema

| Setting              | Default                                                |
| -------------------- | ------------------------------------------------------ |
| Search index         | `{REDIS_NAMESPACE}:idx:agent-memory`                   |
| RedisJSON key prefix | `{REDIS_NAMESPACE}:agent-memory`                       |
| Vector algorithm     | `FLAT`                                                 |
| Vector field         | 512-dimensional `FLOAT32`, cosine distance             |
| Embedding model      | `text-embedding-3-small` with 512 requested dimensions |
| Duplicate threshold  | Cosine distance ≤ `0.1`                                |
| Retrieval threshold  | Cosine distance ≤ `0.3`                                |
| Retrieval limit      | 5 memories by default                                  |
| Exact-match tags     | `memory_type`, `user_id`, `thread_id`, `memory_id`     |
| Retention            | Persistent; no TTL is assigned                         |

Each record stores:

```text
memory_id     Stable ULID used for retrieval and deletion
content       Human-readable memory text
memory_type   episodic | semantic
user_id       Ownership and retrieval boundary
thread_id     Originating conversation, when available
created_at    UTC creation timestamp
metadata      Caller metadata plus the provenance envelope
embedding     512-dimensional semantic vector
```

`episodic` memories represent personal preferences or experiences, such as a preference for
aisle seats. `semantic` memories represent reusable travel knowledge, such as a visa rule. The
category describes intended use; it does not independently verify whether the stored content is
true.

## Runtime defaults

| Setting                         | Value                                      |
| ------------------------------- | ------------------------------------------ |
| Agent model                     | `gpt-5.6-luna`, temperature `0.7`          |
| Summarizer model                | `gpt-5.6-luna`, temperature `0.3`          |
| Model-backed node attempts      | 2                                          |
| Embedding retries               | Maximum 2 with a 10-second request timeout |
| Summarization threshold         | 6 messages                                 |
| Graph recursion limit           | 12                                         |
| Redis connect / command timeout | 2 seconds / 5 seconds                      |

## Simplified Request flow

```text
1. CLI sends one new HumanMessage with user_id and thread_id in RunnableConfig
2. RedisSaver restores the unexpired checkpoint for thread_id
3. Agent receives the travel system prompt plus restored messages
   ├─ memory tool selected
   │  4. Tool reads identity from trusted runtime configuration
   │  5. Redis stores, retrieves, or explicitly deletes user-scoped LTM
   │  6. Tool result returns to the agent
   │  └─ agent may call another tool or produce a response
   └─ no tool selected
4. Summarizer checks the message-count threshold
5. RedisSaver persists the resulting state with the STM expiry policy
6. CLI prints the latest assistant response
```

For detailed sequence diagram depicting the end-to-end Memory request flow, refer to the [memory_sequence_diagram.md](agentic/Memory/memory_sequence_diagram.md) and preview it in https://mermaid.live/

## Path Summary

| Path                    | When                                                     | Terminal Node                                   |
| ----------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| **Direct response**     | Agent answers without calling any memory tool            | `agent → summarize → END`                       |
| **Store memory**        | Agent stores an episodic or semantic memory via LTM tool | `agent → tools → agent → summarize → END`       |
| **Retrieve memories**   | Agent retrieves relevant memories for context            | `agent → tools → agent → summarize → END`       |
| **Delete memory**       | Agent deletes a user-scoped memory by ID                 | `agent → tools → agent → summarize → END`       |
| **Multi-tool turn**     | Agent chains multiple tool calls before responding       | `agent → tools → agent → ... → summarize → END` |
| **Summarize & compact** | Message count ≥ 6 triggers conversation compaction       | `summarize` replaces history with summary       |

## Key Participants

| Participant  | Implementation                                                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Agent        | [`call_agent`](agentic/Memory/agentic_memory.py#L438-L440) — `model_with_tools.invoke([TRAVEL_SYSTEM_PROMPT, ...messages])` |
| ToolNode     | [`ToolNode`](agentic/Memory/agentic_memory.py#L472) — wraps `store_memory`, `retrieve_memories`, `delete_memory`            |
| Store        | [`store_memory`](agentic/Memory/agentic_memory.py#L341-L362) — dedup check + provenance + embed + index.load                |
| Retrieve     | [`retrieve_memories`](agentic/Memory/agentic_memory.py#L365-L384) — user-scoped VectorRangeQuery (distance ≤ 0.3)           |
| Delete       | [`delete_memory`](agentic/Memory/agentic_memory.py#L387-L396) — user-scoped FilterQuery + drop_keys                         |
| Repository   | [`MemoryRepository`](agentic/Memory/agentic_memory.py#L138-L334) — LTM storage, dedup, retrieval, deletion                  |
| Summarizer   | [`summarize_conversation`](agentic/Memory/agentic_memory.py#L442-L467) — RemoveMessage compaction above threshold           |
| Checkpointer | [`RedisSaver`](agentic/Memory/agentic_memory.py#L510-L517) — STM checkpoint with sliding TTL                                |

## Run it

Prerequisites:

- Python 3.13 or later.
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.
- A local Redis 8 instance with Search and JSON commands available.
- An OpenAI API key for generation, summarization, and embeddings.

From the repository root:

The commands below use `uv` directly. `make setup`, `make doctor`, and `make verify` are optional
aliases. `make redis-start` is an optional Homebrew-oriented launcher for the already-installed
Redis server; you may use your normal service manager instead.

```bash
# Install the locked dependencies.
uv sync --locked

# Create local configuration, then add OPENAI_API_KEY to .env.
cp .env.example .env

# Optional: start Redis with the repository's Homebrew-oriented wrapper.
make redis-start

# Verify Python, configuration, Redis connectivity, Search, and JSON support.
uv run portfolio-doctor

# Start the interactive assistant.
uv run python agentic/Memory/agentic_memory.py
```

The CLI asks for both identifiers before starting:

```text
Enter a user ID (default: demo-user): alex
Enter a thread ID (default: demo-thread): london-2026
Connecting to Redis at redis://localhost:6379/0
Welcome to the Travel Assistant! Type 'exit' or 'quit' to stop.
```

The shared `.env` supports these settings:

| Variable                                        | Required | Default / behavior                                       |
| ----------------------------------------------- | -------- | -------------------------------------------------------- |
| `OPENAI_API_KEY`                                | Yes      | No default; required before application initialization   |
| `OPENAI_MODEL`                                  | No       | `gpt-5.6-luna`                                           |
| `OPENAI_EMBEDDING_MODEL`                        | No       | `text-embedding-3-small`                                 |
| `REDIS_URL`                                     | No       | Takes precedence over individual Redis connection fields |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`          | No       | `localhost`, `6379`, `0`                                 |
| `REDIS_USERNAME`, `REDIS_PASSWORD`, `REDIS_SSL` | No       | Optional authentication and TLS settings                 |
| `REDIS_NAMESPACE`                               | No       | `portfolio`                                              |
| `STM_TTL_MINUTES`                               | No       | `1440`                                                   |
| `STM_REFRESH_TTL_ON_READ`                       | No       | `true`; refreshes checkpoint TTL when read               |

## Expected behavior

Model wording and tool selection vary by run, but this sequence exercises both memory layers:

```text
You: Remember that I prefer direct flights and aisle seats.
Assistant: I'll remember that you prefer direct flights and aisle seats.

You: Plan a flight to Lisbon that fits my preferences.
Assistant: I'll prioritize direct options and aisle seating for your Lisbon trip.

You: Forget my aisle-seat preference.
Assistant: I've removed that preference from long-term memory.
```

- Restart with the same `user_id` and `thread_id` before the STM TTL expires to resume the
  checkpointed conversation.
- Start a different thread with the same user to demonstrate fresh STM with reusable LTM.
- Start with a different user and a unique thread ID to demonstrate LTM isolation.
- Exit with `exit` or `quit`; the Redis connection closes while checkpoints and LTM remain
  subject to their respective retention policies.

## Retention, trust, and limitations

The STM expiry and persistent LTM policies are chosen to contrast Redis memory lifecycles in a
working demonstration. They are not recommended production retention periods, consent rules, or
assurances that stored or retrieved memories are accurate, authorized, or safe to reuse.

1. `user_id` and `thread_id` are entered directly in the terminal. They demonstrate partitioning
   but do not authenticate or authorize a real person.
2. `RedisSaver` identifies STM by thread. Reusing a thread ID for another user can expose that
   checkpoint, so production thread IDs must be globally unique and access-controlled.
3. The model decides when to call memory tools. Memory extraction, classification, retrieval,
   and deletion are therefore model-dependent rather than deterministic business rules.
4. Conversation summaries are lossy. A preference that must survive STM expiry should be stored
   explicitly in LTM rather than relying on summarization.
5. LTM retrieval spans a user's threads by default. Duplicate detection also includes the
   current thread when one is supplied, so equivalent memories can exist in different threads.
6. The store operation prevents near-duplicates but does not update or reconcile conflicting
   memories. A correction currently requires deletion followed by a new store.
7. `delete_memory` removes one LTM record; it does not delete STM checkpoints. STM follows its
   independent expiry policy.
8. Stored memory content is returned to the model as tool output. A production system should
   validate it and treat it as untrusted data before including it in model context.
9. The demonstration does not provide encryption, an audit log, or a consent-management layer.
   Durable personal data needs those controls before production use.

## Test it

The focused tests use fakes and an in-memory checkpointer, so they do not contact Redis or
OpenAI:

```bash
uv run python -m unittest \
  tests.test_phase2.MemoryRepositoryTests \
  tests.test_phase2.TravelGraphTests -v
```

With local Redis running, verify durable LTM, scoped deletion, and STM expiry:

```bash
uv run python -m unittest \
  tests.test_redis_integration.RedisIntegrationTests.test_reopening_memory_index_preserves_long_term_memory \
  tests.test_redis_integration.RedisIntegrationTests.test_stm_checkpoint_keys_have_a_sliding_ttl -v
```

Run the complete repository quality gate directly with:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench
```

`make verify` is the optional convenience alias for these commands.
See the repository [test strategy](../../TESTING.md) for how this real persistence layer fits the
fast/Redis/live test pyramid.

## License

This project is available under the repository's [MIT License](../../LICENSE).
