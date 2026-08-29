# Test strategy

The repository uses a test pyramid: most checks are deterministic and isolated, while a smaller
boundary layer executes real Redis commands and an explicitly enabled live layer contacts source
websites and OpenAI.

## Layers

| Layer | Default suite | External resources | Purpose |
| --- | --- | --- | --- |
| Fast behavior | Yes | None | Routing, retry bounds, authorization, cache policy, metrics, evaluation collection, configuration, and API request construction |
| Redis integration | Yes | Local Redis 8 | Search schemas and queries, RedisJSON/Hash serialization, TTL, migrations, checkpoint persistence, scoped cleanup, and all three vector-search demos |
| Live website | No; explicit opt-in | Three configured Flex RAG source pages | Detect source availability, loader/parser changes, and missing source metadata |
| Live OpenAI + Redis | No; explicit opt-in | One embedding request, one bounded GPT-5.6 Luna response, local Redis | Prove the real OpenAI adapter, usage parsing, cache persistence, TTL, and exact-hit reuse together |

The default suite contains 57 fast tests and 14 real-Redis tests: approximately an 80/20 split.
Live tests are excluded from that ratio because they are
deliberately opt-in, metered, and dependent on services outside the repository.

The Redis tests never replace Redis with an in-memory fake. They use unique, colon-separated
namespaces, bounded clients, `try/finally` cleanup, and `SCAN`/`UNLINK` for owned keys. The vector
tests replace only the expensive local embedding models with deterministic vectors; schema
creation, Hash serialization, KNN/range/text/hybrid/aggregation queries, and index deletion all
execute in Redis.

## Run the layers directly

Install the locked environment first:

```bash
uv sync --locked
```

Run the fast layer without Redis or credentials:

```bash
uv run python -m unittest -v \
  tests.test_config \
  tests.test_phase2 \
  tests.test_phase2_requirements \
  tests.test_message_history.MessageHistoryUnitTests \
  tests.test_semantic_cache \
  tests.test_workbench.WorkbenchUnitTests
```

With Redis 8 running locally, run only the real-Redis layer:

```bash
uv run python -m unittest -v \
  tests.test_message_history.MessageHistoryRedisIntegrationTests \
  tests.test_redis_integration \
  tests.test_semantic_cache_redis \
  tests.test_vector_search_redis \
  tests.test_workbench.WorkbenchRedisIntegrationTests
```

The complete default suite runs both layers. Redis-dependent classes report a skip when Redis,
Search, or JSON is unavailable; CI provides Redis 8 and therefore executes them:

```bash
uv run python -m unittest discover -s tests -v
```

`make test-fast`, `make test-integration`, and `make verify` are optional aliases for these
repository commands.

## Run live integrations intentionally

Live tests never run merely because an API key exists. Each requires an explicit flag.

Check the three source websites used by Flex RAG:

```bash
RUN_LIVE_WEB_TESTS=1 \
  uv run python -m unittest tests.test_live_integrations.LiveSourceWebsiteTests -v
```

Check one cold OpenAI-backed semantic-cache request followed by an exact Redis hit:

```bash
RUN_LIVE_OPENAI_TESTS=1 \
  uv run python -m unittest tests.test_live_integrations.LiveOpenAIRedisTests -v
```

The OpenAI test requires `OPENAI_API_KEY` and local Redis 8. It makes at most one embedding call
and one response call, caps generation at 96 output tokens, disables provider-side response
storage, validates token usage, and deletes its unique Redis index and documents afterward.

To run both live classes:

```bash
RUN_LIVE_INTEGRATIONS=1 uv run python -m unittest tests.test_live_integrations -v
```

Equivalent optional aliases are `make test-live-web`, `make test-live-openai`, and `make
test-live`. Because source availability and model access can change independently of a commit,
the live layer is available through a manually dispatched GitHub Actions workflow rather than
the required pull-request gate.

## Coverage by example

| Example | Fast behavior | Real Redis boundary | Live external boundary |
| --- | --- | --- | --- |
| Semantic cache | Cache-aside decisions, guards, metrics, OpenAI request adapter | Exact/semantic hits, JSON, TTL, partitioning, invalidation | OpenAI embedding + GPT-5.6 Luna + Redis cold/exact flow |
| Flex RAG | Routing, rewrite bounds, relevance and injection boundaries | Persisted-index behavior is covered through shared Redis/Search contracts | All configured source URLs through the example's real loader |
| STM/LTM memory | Provenance and scoped deletion | RedisSaver persistence, checkpoint TTL, durable memory reopening | Not needed for the persistence boundary |
| Role-based RAG | Role filters, citations, grounded-answer rules | Citation-schema migration without data loss | Full PDF embedding/model evaluation remains an intentional manual demo |
| Retrieval evaluation | Retrieve-once generation/scoring and complete Batch collection | Temporary index lifecycle uses the shared Redis integration contract | Batch/Ragas runs remain explicit experiments because their cost scales with the dataset |
| Message history | Role mapping, OpenAI payloads, session orchestration, factory wiring, and failure cleanup | Dedicated RedisVL session isolation, ordering, Hash serialization, clear, and index deletion tests | The runnable demo provides the intentional multi-call OpenAI path |
| redis-py vector search | Query helpers | Complete demo: Hashes, FLAT KNN, filters, range, BM25, aggregation | Local model download is intentionally replaced with fixed vectors in tests |
| RedisVL vector search | Declarative query construction | Complete demo: ingestion, KNN, filters, range, text, hybrid, cleanup | Local model download is intentionally replaced with fixed vectors in tests |
| Multi-vector search | Vector shape/dtype contract | Complete demo: three HNSW fields and weighted multi-vector query | Local model downloads are intentionally replaced with fixed vectors in tests |
| Workbench | Event replay, deterministic mode, browser contract, mocked adapter | Four request paths, sanitized inspector, sensitive-data non-retention | The real cache adapter is exercised by the live OpenAI + Redis flow |

The deliberately manual rows are not disguised as automated coverage: full PDF ingestion,
OpenAI Batch evaluation, Ragas judging, and the three-call message-history walkthrough can incur
material cost or vary with a third-party model. They remain runnable examples, while the bounded
live smoke tests prove the shared network and model adapters at predictable cost.
