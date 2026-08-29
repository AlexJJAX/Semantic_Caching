# RedisVL Multi-Session Message History with GPT-5.6 Luna

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Tests](https://img.shields.io/badge/focused%20tests-9%20passing-2E7D32)
![Integration](https://img.shields.io/badge/Redis%20integration-2%20passing-00796B)
![Model](https://img.shields.io/badge/model-gpt--5.6--luna-412991)
![Redis](https://img.shields.io/badge/Redis-8%20Search-DC382D)
![Context](https://img.shields.io/badge/context-session--scoped%20recency-00796B)
![Lifecycle](https://img.shields.io/badge/lifecycle-ephemeral-E65100)
![License](https://img.shields.io/badge/license-MIT-455A64)

A controlled demonstration of three independent conversation histories stored in one RedisVL
message index. Each logical session receives the same financial-planning prompt, but GPT-5.6
Luna receives only the history selected by that session's `session_tag`.

The example isolates context in Redis, adapts RedisVL message roles to the OpenAI Chat
Completions schema, stores each new exchange, and removes its complete demo namespace on exit.
It is a generic, demonstrational working primitive intended to showcase Redis-backed session
partitioning within one process. It is not intended or suitable for production use or as a durable
chat application; the personas, session tags, context window, model calls, and cleanup policy are
fixed to make the storage and retrieval lifecycle easy to inspect.

## Architecture Overview

| Component                                               | Responsibility                                                                                                    |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `SemanticMessageHistory`                                | Stores all message records in one Redis Search index and partitions them with `session_tag`                       |
| Redis session filter                                    | Selects a single session before sorting messages by timestamp                                                     |
| Local vectorizer                                        | Embeds every stored message with `sentence-transformers/all-mpnet-base-v2`                                        |
| `OpenAIClient.remap()`                                  | Preserves current `assistant` roles, accepts legacy RedisVL `llm` records, and rejects unknown roles             |
| `OpenAIClient.converse()`                               | Sends the selected history plus the new prompt to GPT-5.6 Luna through Chat Completions                           |
| `create_message_history()`                              | Builds the namespaced RedisVL index and accepts an injected vectorizer for deterministic boundary tests           |
| `store_exchange()`                                      | Stores a user/assistant pair using RedisVL's current role convention under one explicit session tag               |
| `run_demo()`                                            | Seeds three personas, asks the same question in each session, stores the exchanges, and prints one session        |
| `main()`                                                | Validates configuration, creates external resources, clears stale demo data, and guarantees scoped cleanup        |
| [Sequence diagram](message_history_sequence_diagram.md) | Traces initialization, session-tagged writes and reads, role mapping, model calls, inspection, and scoped cleanup |

All executable behavior lives in [`Multiple_sessions.py`](./Multiple_sessions.py). Importing the
module loads shared configuration but does not connect to Redis or call OpenAI.

### End-to-end flow sequence diagram

For the complete interaction among the application, RedisVL, the local vectorizer, Redis, and
the model, see the
[multi-session message-history sequence diagram](message_history_sequence_diagram.md).

## Controlled demonstration

The script uses one history object and one model while changing only the `session_tag` and its
stored context:

| Session tag          | Seeded context                                        | Expected emphasis                               |
| -------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| `student`            | Part-time income with rent, utilities, and groceries  | Cash-flow discipline and emergency savings      |
| `young professional` | Higher income, city expenses, existing emergency fund | Retirement contributions and long-term planning |
| `retired pensioner`  | Pension income, owned home, and lower recurring costs | Income preservation and retirement budgeting    |

Every session receives this exact prompt:

```text
What is the single most important thing I should focus on financially?
```

The model's different responses provide a visible test of whether the intended session history
was supplied. The sessions are processed sequentially—not concurrently—and the wording remains
model-dependent.

## What it demonstrates

- Multiple logical conversations partitioned inside one Redis Search index.
- Exact `session_tag` filtering before context reaches the model.
- Timestamp-ordered recovery of the five most recent messages by default.
- A controlled same-prompt, different-context comparison.
- Local Hugging Face embeddings for every stored message.
- RedisVL-to-OpenAI role adaptation.
- Appending each prompt/response pair to its originating session.
- Lowercase, colon-separated Redis names from the shared namespace configuration.
- Bounded Redis connection and command timeouts through the shared client factory.
- Scoped index/key cleanup and deterministic client closure on success or failure.

## Key Design Decisions

- **Use one index for all sessions** — `session_tag` is an indexed Redis TAG field, allowing one
  message-history abstraction to serve several independent conversations.
- **Filter at retrieval time** — `get_recent(session_tag=...)` issues a Redis query scoped to the
  requested session. The application does not fetch all messages and partition them in Python.
- **Use recency for the demo** — the context path sorts by timestamp and returns at most five
  messages. Although RedisVL stores vectors and supports `get_relevant()`, this script does not
  use semantic history retrieval.
- **Keep the experiment controlled** — the same prompt and same model are used for all three
  sessions, making stored history the principal input difference.
- **Map roles explicitly** — the adapter preserves current `assistant`, `user`, and `system`
  roles, maps legacy RedisVL `llm` records to `assistant`, and fails on unrecognized roles.
- **Store complete exchanges** — after each model response, `store_exchange()` appends both the
  user prompt and the `assistant` response under the same session tag.
- **Reset before seeding** — startup calls `history.clear()` so stale messages from an interrupted
  prior run cannot change the controlled comparison.
- **Make the demo ephemeral** — after initialization enters the guarded run block, shutdown calls
  `history.delete()`, dropping the owned Search index and its message keys while preserving
  unrelated Redis data.

## Session isolation contract

```text
session_tag supplied by application
             ↓
Redis TAG filter: @session_tag == requested tag
             ↓
sort matching records by timestamp descending
             ↓
take top 5 and restore chronological order
             ↓
map roles → append new prompt → OpenAI
```

The three tags are labels, not user accounts. Isolation depends on the application always
supplying the correct tag; this script has no authentication or authorization layer.

## Redis storage model

| Setting               | Value                                                                        |
| --------------------- | ---------------------------------------------------------------------------- |
| Search index          | `{REDIS_NAMESPACE}:idx:message-history:budgeting`                            |
| Message key prefix    | `{REDIS_NAMESPACE}:message-history:budgeting:`                               |
| Storage type          | Redis Hash                                                                   |
| Context selector used | `get_recent(session_tag=...)`                                                |
| Context limit         | 5 messages by default                                                        |
| Ordering              | Timestamp descending in Redis, reversed to chronological order for the model |
| Vectorizer            | Local `sentence-transformers/all-mpnet-base-v2`                              |
| Vector index          | `FLAT`, `FLOAT32`, cosine distance; dimensions inferred from the vectorizer  |
| Semantic threshold    | Cosine distance `0.3` by default, unused by the demonstrated recency path    |
| TTL                   | None                                                                         |
| Retention             | Cleared at startup and deleted at shutdown                                   |

RedisVL creates these indexed fields:

```text
entry_id      Stable message-record identifier
session_tag   Exact session partition
role          system | user | assistant | tool
content       Message text
timestamp     Numeric ordering field
tool_call_id  Optional tool correlation
metadata      Optional serialized metadata
vector_field  Local semantic embedding
```

The script does not use `OPENAI_EMBEDDING_MODEL`. Message embeddings are computed locally by
RedisVL's default Hugging Face vectorizer, while `OPENAI_MODEL` controls chat generation.

## Request flow

```text
1. Validate OPENAI_API_KEY and connect to local Redis
2. Create or validate the namespaced SemanticMessageHistory index
3. Clear any stale keys owned by this demonstration
4. Seed four messages under each of three session tags
5. For each session, sequentially:
   a. Retrieve only that session's recent messages
   b. Preserve assistant/user/system roles and convert legacy llm → assistant
   c. Append the shared user prompt
   d. Call GPT-5.6 Luna through OpenAI Chat Completions
   e. Store the new prompt and response under the same session tag
   f. Print the response
6. Retrieve and print the five most recent student messages
7. Delete the owned index and messages
8. Close OpenAI and Redis clients
```

The initial model call for each persona receives all four seeded messages. After the new exchange
is stored, that session contains six messages; the final `get_recent()` inspection prints only
the newest five because it uses RedisVL's default limit.

## Run it

Prerequisites:

- Python 3.13 or later.
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.
- A local Redis 8 instance with Search and JSON commands available.
- An OpenAI API key for the three live chat-completion calls.
- Network access on first use if the local Hugging Face vectorizer is not already cached.

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

# Validate the runtime directly.
uv run portfolio-doctor

# Run the three-session comparison.
uv run python llm_message_history/Multiple_sessions.py
```

The shared `.env` settings used by this script are:

| Variable                                        | Required | Default / behavior                                       |
| ----------------------------------------------- | -------- | -------------------------------------------------------- |
| `OPENAI_API_KEY`                                | Yes      | No default; checked before connecting to Redis           |
| `OPENAI_MODEL`                                  | No       | `gpt-5.6-luna`                                           |
| `REDIS_URL`                                     | No       | Takes precedence over individual Redis connection fields |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`          | No       | `localhost`, `6379`, `0`                                 |
| `REDIS_USERNAME`, `REDIS_PASSWORD`, `REDIS_SSL` | No       | Optional authentication and TLS settings                 |
| `REDIS_NAMESPACE`                               | No       | `portfolio`                                              |

There are no command-line flags; personas, seed histories, shared prompt, and recent-message
limit are intentionally fixed in the script.

## Expected output

Model wording varies, but the terminal output follows this structure:

```text
Student:  What is the single most important thing I should focus on financially?

LLM:  ...student-specific budgeting response...
Young Professional:  What is the single most important thing I should focus on financially?

LLM:  ...retirement-focused response...
Retiree:  What is the single most important thing I should focus on financially?

LLM:  ...fixed-income retirement response...

Student session history:
{'role': 'user', 'content': '...'}
{'role': 'assistant', 'content': '...'}
...
```

After normal completion, the demo's index and message keys no longer exist. This cleanup is
intentional; the example proves within-run partitioning rather than restart persistence.

## Trust, lifecycle, and limitations

The lifecycle deliberately proves create, isolate, retrieve, and clean up behavior for a Redis
message-history primitive. It does not provide the identity, authorization, retention, recovery,
moderation, or operational controls required by a production conversation service.

1. `session_tag` is a routing value, not an authorization control. A caller that can choose
   another session's tag can request that session's context.
2. The three histories are processed sequentially. The script does not measure concurrency,
   connection-pool behavior, throughput, or race conditions.
3. Context selection is recency-based and limited to five messages. There is no summarization,
   token-budget calculation, or automatic preservation of an older system prompt.
4. Every message is embedded locally even though the example does not call semantic retrieval.
   This demonstrates the RedisVL semantic-history schema but adds model download, CPU, storage,
   and ingestion overhead to a recency-only workflow.
5. Stored roles and content are trusted when reconstructed. A production service must prevent
   users from injecting privileged `system` messages or writing directly to another session.
6. No TTL is configured. `finally` cleanup handles ordinary exceptions, but a forced process
   termination can leave data behind until the next startup clears the owned namespace.
7. Runs sharing the same `REDIS_NAMESPACE` and fixed index name will clear or delete one another's
   data. Use an isolated namespace before executing concurrent copies.
8. Conversation histories and prompts are sent to OpenAI. Apply consent, minimization,
   redaction, encryption, and retention controls before storing real personal or financial data.
9. The generated responses are an architectural demonstration, not financial advice or a
   validated budgeting system.

## Test it

Nine dedicated unit tests use injected histories and mocked OpenAI clients. They validate current
and legacy role mapping, Chat Completions payload construction, empty-response handling, three-way
session orchestration, namespaced factory wiring, and cleanup across success and failure paths
without contacting Redis, downloading a model, or consuming OpenAI calls:

```bash
uv run python -m unittest \
  tests.test_message_history.MessageHistoryUnitTests -v
```

Two integration tests use a deterministic four-dimensional vectorizer but a real local RedisVL
index and Redis Search. They prove exact `session_tag` isolation, timestamp ordering, Hash
serialization, session counts, scoped clear, index deletion, and preservation of unrelated keys:

```bash
uv run python -m unittest \
  tests.test_message_history.MessageHistoryRedisIntegrationTests -v
```

Run the repository-wide quality gate directly with local Redis available:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench
```

`make verify` is the optional convenience alias for these commands.

The live command is the end-to-end integration check: it initializes the local vectorizer,
writes and retrieves all three Redis sessions, makes three OpenAI calls, and then removes its
owned state.

The local model download and three OpenAI calls remain an intentional manual boundary; the
dedicated automated module covers the application logic and real Redis persistence separately.
See the repository [test strategy](../TESTING.md) for the complete test pyramid.

## License

This project is available under the repository's [MIT License](../LICENSE).
