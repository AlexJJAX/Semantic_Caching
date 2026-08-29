# Redis AI Engineering Portfolio

[![Run unit Tests | Passing](https://github.com/AlexJJAX/Semantic_Caching/actions/workflows/unit-tests.yml/badge.svg?branch=master)](https://github.com/AlexJJAX/Semantic_Caching/actions/workflows/unit-tests.yml)
![Tests](https://img.shields.io/badge/tests-unit%20%2B%20integration-2E7D32)
![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Redis](https://img.shields.io/badge/Redis-8%20Search%20%7C%20JSON-DC382D)
![Model](https://img.shields.io/badge/default%20model-gpt--5.6--luna-412991)
![License](https://img.shields.io/badge/license-MIT-455A64)

A runnable collection of Redis patterns for modern AI applications: semantic caching,
short- and long-term memory, retrieval-augmented generation, authorization-aware retrieval,
evaluation, and vector search.

Every example is a generic, demonstrational working primitive intended to make one Redis
capability for AI applications concrete and inspectable. Collectively they cover semantic
caching, short-term memory (STM), long-term memory (LTM), RAG, evaluation, and vector retrieval.
They are not intended or suitable for production use; runnable behavior and integration tests do
not imply production readiness, security, reliability, governance, or operational completeness.

The examples progress from low-level `redis-py` queries to application-shaped LangChain and
LangGraph workflows. They share one locked `uv` environment, one typed configuration layer,
scoped Redis names, bounded clients, and a local Redis 8 runtime installed with Homebrew.
GPT-5.6 Luna is the default generative model; the foundational vector examples run with local
Sentence Transformers.

## Portfolio at a glance

| Area                                                                | What it demonstrates                                                                                                                                          | Redis capabilities                                                                               | Start here                                                                                                              |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| [Redis AI Workbench](workbench/README.md)                           | Four interactive demonstrations with a live request-processing timeline, metrics, comparison tables, and a sanitized state inspector                          | RedisJSON, TTL, Search, permission filters, key/index metadata, SSE-backed traces                | `uv run python workbench/server.py`                                                                                     |
| [Semantic cache](semantic_cache/README.md)                          | Exact-first and semantic-second cache-aside flow with strict partitions, bypass policy, false-hit guards, calibration, invalidation, and cost/latency metrics | Direct JSON reads, filtered vector range search, transactional TTL writes, TAG/SCAN invalidation | [`semantic_cache_demo.py`](semantic_cache/semantic_cache_demo.py)                                                       |
| [Bounded agentic RAG](agentic/Flex_rag/README.md)                   | Model-directed retrieval, relevance grading, real query rewrites, bounded retries, grounded generation, and citations                                         | Persistent vector index, threshold retrieval, source metadata                                    | [`Langgraph_redis_agentic_flex_rag.py`](agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py)                           |
| [STM/LTM travel assistant](agentic/Memory/README.md)                | Expiring conversation checkpoints alongside durable episodic and semantic memories with provenance and explicit deletion                                      | `RedisSaver`, RedisJSON vector index, user/thread TAG filters, sliding STM TTL                   | [`agentic_memory.py`](agentic/Memory/agentic_memory.py)                                                                 |
| [Role-based RAG](RAG/README.md)                                     | Authorization enforced inside retrieval so unauthorized passages never become model context; grounded answers include source/page citations                   | RedisJSON users/documents, `allowed_roles` TAG filters, vector range queries                     | [`User_role_based_rag.py`](RAG/User_role_based_rag.py)                                                                  |
| [Multi-session message history](llm_message_history/README.md)      | Three session-tagged histories receiving the same prompt, with recency retrieval and RedisVL-to-OpenAI role mapping                                           | One Hash/Search index, exact session TAG filtering, timestamps, stored local vectors             | [`Multiple_sessions.py`](llm_message_history/Multiple_sessions.py)                                                      |
| [RAG evaluation](evaluation/README.md)                              | OpenAI Batch document embeddings, reproducible manifests, partial collection, retrieve-once generation/scoring, and four Ragas metrics                        | Temporary vector index, threshold retrieval, scoped cleanup                                      | [`generate_testset.py`](evaluation/generate_testset.py) · [`06_ragas_evaluation.py`](evaluation/06_ragas_evaluation.py) |
| [Vector search with redis-py](vector_search/1_redispy/README.md)    | Native schema creation plus KNN, vector-range, TAG, numeric, full-text, BM25, and aggregation queries                                                         | Redis Hashes, `FT.CREATE`, Search query syntax, `FLAT` vectors                                   | [`Redispy.py`](vector_search/1_redispy/Redispy.py)                                                                      |
| [Vector search with RedisVL](vector_search/2_redisvl/README.md)     | Declarative schemas, DataFrame ingestion, embedding cache, typed filters, range search, BM25, and hybrid scoring                                              | `SearchIndex`, `VectorQuery`, `RangeQuery`, `TextQuery`, `AggregateHybridQuery`                  | [`Redisvl.py`](vector_search/2_redisvl/Redisvl.py)                                                                      |
| [Multi-vector search](vector_search/3_multivector_search/README.md) | Weighted retrieval across three embedding models with different dimensions and numeric precision                                                              | Three HNSW vector fields, isolated embedding caches, `MultiVectorQuery` score blending           | [`Multivector_search.py`](vector_search/3_multivector_search/Multivector_search.py)                                     |

Each linked README explains the example's architecture, data model, commands, expected behavior,
retention policy, tests, and limitations.

## Redis AI Workbench

The fastest review path is the local Workbench. Its four tabs cover semantic cache, STM/LTM,
RBAC RAG, and retrieval evaluation around one visible lifecycle:

```text
Prompt → cache decision → retrieval → model → STM/LTM write → metrics
```

Live mode is the default. It performs real local Redis operations and calls GPT-5.6 Luna when a
scenario reaches its model stage. An explicit `demo` mode provides repeatable local responses
for offline review; it never silently replaces a failed live request.

The browser receives only sanitized operational state. The inspector exposes key names, Redis
types, TTLs, memory sizes, and index counts—not prompts, answers, permissions, or embeddings.

## Quick start

Prerequisites:

- Python 3.13 or later.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- A local Homebrew installation of Redis 8 with Search and JSON commands.
- An OpenAI API key for live model and OpenAI-embedding examples.

From the repository root:

`make` is optional. It provides memorable aliases for repository workflows; `uv` remains the
Python environment and execution tool. The only Homebrew-oriented alias is `make redis-start`,
which launches the already-installed Redis server with repository-local data, PID, and log files.
You may start Redis with your usual service manager instead.

```bash
# Install the exact versions recorded in uv.lock.
uv sync --locked

# Create local configuration and add OPENAI_API_KEY.
cp .env.example .env

# Optional: start Redis with the repository's Homebrew-oriented wrapper.
make redis-start

# Validate Python, configuration, Redis, Search, and JSON directly.
uv run portfolio-doctor

# Start the Workbench at http://127.0.0.1:8123.
uv run python workbench/server.py --host 127.0.0.1 --port 8123
```

If port `8123` is occupied:

```bash
uv run python workbench/server.py --host 127.0.0.1 --port 8124
```

For the explicit simulator:

```bash
WORKBENCH_MODEL_MODE=demo uv run python workbench/server.py --host 127.0.0.1 --port 8123
```

Equivalent optional aliases are `make setup`, `make doctor`, and `make workbench`. They delegate to
the direct commands above; `make` does not install or replace `uv`.

Run individual scripts from the repository root because their resource paths are relative to
that location. For example:

```bash
uv run python semantic_cache/semantic_cache_demo.py benchmark --iterations 3
uv run python agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py
uv run python agentic/Memory/agentic_memory.py
uv run python RAG/User_role_based_rag.py
```

## Shared configuration

All examples load the repository-root `.env` through
[`PortfolioSettings`](src/redis_ai_portfolio/config.py). The core settings are:

```bash
OPENAI_API_KEY="sk-your-openai-api-key"
OPENAI_MODEL="gpt-5.6-luna"
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

# REDIS_URL takes precedence when set.
REDIS_URL="redis://localhost:6379/0"
REDIS_NAMESPACE="portfolio"

WORKBENCH_MODEL_MODE="live"
CACHE_DISTANCE_THRESHOLD="0.20"
CACHE_TTL_SECONDS="3600"
STM_TTL_MINUTES="1440"
STM_REFRESH_TTL_ON_READ="true"
```

See [`.env.example`](.env.example) for individual Redis host, port, database, authentication,
TLS, retention, and cost-estimation settings. Credentials are loaded server-side and are never
sent to the Workbench browser.

## Shared engineering contract

- **Redis before the model** — exact cache keys, retrieval thresholds, and authorization filters
  reduce what reaches the LLM rather than relying only on prompt instructions.
- **Explicit retention** — temporary checkpoints and cache entries expire; durable memories
  retain provenance and require scoped deletion; intentionally ephemeral demos remove only their
  own keys and indexes.
- **Evidence boundaries** — grounded RAG branches preserve source/page metadata, delimit
  retrieved text as untrusted data, cite supporting evidence, and include insufficient-context
  responses.
- **Fair evaluation** — every evaluation row retrieves once and passes the identical documents
  to answer generation and Ragas scoring.
- **Bounded operations** — shared Redis clients use connection/command timeouts and limited
  retries; agent loops also cap node retries, rewrites, and recursion.
- **Safe coexistence** — names are lowercase and colon-separated under `REDIS_NAMESPACE`.
  Cleanup is example-scoped and never flushes the database.
- **Import-safe construction** — agent and cache applications initialize external clients and
  indexes through explicit factories, making core behavior testable with fakes.
- **Layered verification** — fast behavior tests are complemented by real Redis query tests and
  explicitly enabled, bounded website/OpenAI smoke tests.
- **Demonstrational scope** — these contracts make Redis behavior visible and testable; they are
  teaching boundaries, not claims that the examples form production applications.

## Repository map

```text
Semantic_Caching/
├── semantic_cache/             # Exact-first cache, benchmark, calibration, invalidation
├── agentic/
│   ├── Flex_rag/               # Bounded, self-correcting LangGraph RAG
│   └── Memory/                 # Redis checkpoint STM + durable RedisVL LTM
├── RAG/                        # Role-filtered retrieval and grounded chat
├── evaluation/                 # Batch-assisted testset generation and Ragas evaluation
├── llm_message_history/        # Session-tagged RedisVL conversation history
├── vector_search/
│   ├── 1_redispy/              # Native Redis Search queries
│   ├── 2_redisvl/              # RedisVL query abstractions and hybrid scoring
│   └── 3_multivector_search/   # Weighted, multi-model vector retrieval
├── workbench/                  # Local HTTP/SSE server and accessible browser interface
├── src/redis_ai_portfolio/     # Shared configuration, clients, cache, and Workbench engine
├── resources/                  # Included PDFs and movie dataset
├── tests/                      # Offline behavior and local Redis integration suites
├── TESTING.md                  # Test pyramid, commands, live gates, and coverage matrix
├── Makefile                    # Setup, Redis lifecycle, Workbench, and quality commands
├── pyproject.toml              # Python metadata and dependency declarations
└── uv.lock                     # Reproducible dependency lock
```

System-level documentation is also available in the generated OpenWiki pages:
[architecture](openwiki/architecture/overview.md), [quick start](openwiki/quickstart.md),
[operations](openwiki/operations.md), and
[RAG/agent workflows](openwiki/workflows/rag-and-agents.md).

## Quality gate

Run the underlying checks directly:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench
```

`make verify` is the optional convenience alias for these three commands. The gate runs Ruff,
offline and local-Redis behavior tests, and bytecode compilation across all application
directories. Redis-dependent tests skip with an explicit reason when the local service is
unavailable; GitHub Actions runs the complete suite against Redis 8.

The default suite remains approximately 80/20: 57 fast isolated tests and 14 tests that execute
real Redis Search, JSON, Hash, TTL, serialization, session isolation, migration, checkpoint, and
cleanup behavior.
Run either layer directly with `make test-fast` or `make test-integration`.

Live tests are separately gated so a normal test run cannot contact source websites or spend API
credits. `make test-live-web` checks all configured Flex RAG pages; `make test-live-openai` makes
one bounded OpenAI-backed cache miss and verifies the following exact hit comes from Redis. See
the complete [test strategy and per-example coverage matrix](TESTING.md).

## Scope and limitations

- Every directory contains a generic working primitive for demonstrating a specific Redis AI
  capability. None is intended or suitable for production deployment without independent
  architecture, security, privacy, reliability, scalability, governance, and operational work.
- Example user IDs, roles, thread IDs, and session tags illustrate partitioning but are not
  authentication or authorization systems.
- Cache bypass and prompt-injection delimiters are defense-in-depth controls, not data-loss
  prevention or a complete content-safety system.
- Measured latency, cost savings, similarity thresholds, and Ragas scores are workload-specific;
  recalibrate and remeasure them against representative production traffic.
- Some examples deliberately retain indexes or memories across runs, while others delete their
  owned state. Review the linked README before using a shared Redis database.
- First-time local embedding examples may download Sentence Transformer models. OpenAI-backed
  examples require network access and may incur provider charges.

## License and attribution

Released under the [MIT License](LICENSE).

This work was adapted and substantially expanded from the
[Redis LLM examples](https://github.com/redis-developer/redis-py-llm-examples).
