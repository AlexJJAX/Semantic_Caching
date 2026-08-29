# Testing and verification

The repository now uses a layered test strategy: deterministic unit tests by default, real Redis Search/JSON integration tests, and explicitly opt-in live website/OpenAI smoke tests. The complete rationale, commands, and per-example coverage matrix live in [`TESTING.md`](../TESTING.md); this page is the wiki navigation point for that contract.

## Test layers

| Layer | Default? | What it proves |
|---|---:|---|
| Fast behavior | Yes | Routing, cache policy, authorization, metrics, request construction, configuration, and failure bounds without external services |
| Redis integration | Yes, when Redis 8 is available | Real Search/JSON/Hash behavior, vector schemas and queries, TTL, persistence, session isolation, migrations, and scoped cleanup |
| Live websites | No; explicit opt-in | The three Flex RAG source pages remain reachable and load with source metadata |
| Live OpenAI + Redis | No; explicit opt-in | One bounded cold semantic-cache request persists an answer and the following exact hit avoids another embedding/generation call |

Redis integration tests use unique namespaces and cleanup owned keys/indexes; vector tests substitute deterministic vectors only for model downloads while executing the Redis queries for real. Live tests are never enabled merely because credentials exist.

## Commands

Install the locked environment:

```bash
uv sync --locked
```

Run fast tests:

```bash
uv run python -m unittest -v \
  tests.test_config \
  tests.test_phase2 \
  tests.test_phase2_requirements \
  tests.test_message_history.MessageHistoryUnitTests \
  tests.test_semantic_cache \
  tests.test_workbench.WorkbenchUnitTests
```

With Redis 8 and its Search/JSON capabilities running, run the integration layer:

```bash
uv run python -m unittest -v \
  tests.test_message_history.MessageHistoryRedisIntegrationTests \
  tests.test_redis_integration \
  tests.test_semantic_cache_redis \
  tests.test_vector_search_redis \
  tests.test_workbench.WorkbenchRedisIntegrationTests
```

The full default suite is `uv run python -m unittest discover -s tests -v`. Redis-dependent tests skip with an explicit reason when the local capabilities are unavailable. `make test-fast`, `make test-integration`, and `make verify` are convenience aliases; `make verify` also runs Ruff and bytecode compilation.

Opt into live checks deliberately:

```bash
RUN_LIVE_WEB_TESTS=1 uv run python -m unittest tests.test_live_integrations.LiveSourceWebsiteTests -v
RUN_LIVE_OPENAI_TESTS=1 uv run python -m unittest tests.test_live_integrations.LiveOpenAIRedisTests -v
# Or both:
RUN_LIVE_INTEGRATIONS=1 uv run python -m unittest tests.test_live_integrations -v
```

The OpenAI smoke test requires `OPENAI_API_KEY` and local Redis 8, makes at most one embedding and one bounded response call, validates usage and exact-hit reuse, and cleans its unique namespace. The manually dispatched [live-integrations workflow](../.github/workflows/live-integrations.yml) provides the CI entry point for these external checks; they are not part of the required pull-request gate.

## Change guidance

- Message-history changes: start with [`tests/test_message_history.py`](../tests/test_message_history.py), which covers role mapping, OpenAI payloads, session partitioning, factory wiring, cleanup, and Redis-backed isolation.
- Vector-search changes: use [`tests/test_vector_search_redis.py`](../tests/test_vector_search_redis.py); it executes all three examples against real Redis while replacing only embedding-model computation.
- Flex RAG/OpenAI adapter changes: inspect [`tests/test_live_integrations.py`](../tests/test_live_integrations.py) and keep external calls explicitly bounded and opt-in.
- Before relying on a change, run the smallest affected layer, then `make verify` when the environment supports it.
