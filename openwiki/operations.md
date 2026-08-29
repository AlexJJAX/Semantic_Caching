# Operations and runbook notes

## Prerequisites

- Python `3.13+` and `uv` (from `pyproject.toml`). Several local READMEs still say `3.10+`; treat that as stale until reconciled.
- Redis Stack or equivalent RediSearch, RedisJSON, and vector-search support.
- `OPENAI_API_KEY` for OpenAI-backed RAG, agents, history, and evaluation.
- Network access for OpenAI calls and the Flex RAG source URLs.
- Repository-root working directory for relative `resources/...` paths.

Install dependencies and run the local preflight with:

```bash
make setup
make redis-start
make doctor
```

`make setup` uses `uv sync --locked`. `make redis-start` creates `.redis-data/` and starts Redis using the Homebrew Redis configuration; adjust `REDIS_CONFIG` when needed.

## Configuration names

Copy `.env.example` to a local, ignored `.env` and replace placeholders outside version control. `PortfolioSettings.from_env()` loads it without overriding variables already supplied by the process:

- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`.
- `REDIS_URL` (when present, it takes precedence over `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_USERNAME`, `REDIS_PASSWORD`, and `REDIS_SSL`).
- `REDIS_NAMESPACE` (default `portfolio`), plus `CACHE_DISTANCE_THRESHOLD` (default `0.2`) and `CACHE_TTL_SECONDS` (default `3600`) for semantic caching.
- `CACHE_LLM_INPUT_COST_PER_MILLION`, `CACHE_LLM_OUTPUT_COST_PER_MILLION`, and `CACHE_EMBEDDING_COST_PER_MILLION` configure the cache's estimated cost metrics; they are estimates, not billing records.
- `WORKBENCH_MODEL_MODE` is `live` by default and accepts only `live` or `demo`; live mode requires `OPENAI_API_KEY`, while demo mode is deterministic and local.
- `STM_TTL_MINUTES` (default `1440`) controls short-term checkpoint expiry; `STM_REFRESH_TTL_ON_READ` (default `true`) selects sliding versus fixed retention. These settings do not expire long-term memories.

The shared layer validates URL scheme, ports, database, namespace, booleans, and numeric bounds. It also redacts credentials when printing a Redis URL. Do not document or paste actual secret values. Environment files in this checkout should be treated as sensitive.

## Runbook by capability

| Goal | Command |
|---|---|
| Redis AI Workbench (live) | `make workbench` |
| Redis AI Workbench (offline simulator) | `make workbench WORKBENCH_MODEL_MODE=demo` |
| Semantic-cache request/benchmark/calibration | `uv run python semantic_cache/semantic_cache_demo.py ask ...` / `benchmark` / `calibrate` |
| Native KNN/hybrid search | `uv run python vector_search/1_redispy/Redispy.py` |
| RedisVL search | `uv run python vector_search/2_redisvl/Redisvl.py` |
| Multi-vector ranking | `uv run python vector_search/3_multivector_search/Multivector_search.py` |
| Role-filtered RAG | `uv run python RAG/User_role_based_rag.py` |
| Flex agentic RAG | `uv run python agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py` |
| Dual-memory agent | `uv run python agentic/Memory/agentic_memory.py` |
| Session history | `uv run python llm_message_history/Multiple_sessions.py` |
| Ragas scoring | `uv run python evaluation/06_ragas_evaluation.py` |

Evaluation testset generation is a separate submit/collect workflow documented in [Search and evaluation](search-and-evaluation.md).

The Workbench binds to `127.0.0.1:8123` by default; override `WORKBENCH_HOST` and `WORKBENCH_PORT` through Make variables. It exposes `/ready`, `/api/status`, `/api/runs`, replayable `/api/runs/{id}/events` SSE, `/api/redis`, and confirmed `DELETE /api/workbench`. Request bodies are capped at 16 KiB, prompts at 2,000 characters, and inspector responses contain only key/index metadata. Reset is scoped to Workbench-owned names and never flushes Redis.

## Safety and lifecycle

- The examples now derive keys and index names from the lowercase `REDIS_NAMESPACE` (default `portfolio`) and clean up their own indexes/keys rather than flushing the whole Redis database. Embedding-cache data may remain where the demo does not explicitly remove it.
- The evaluation tears down its namespaced index after scoring.
- Memory startup reuses its long-term index with `overwrite=False`; LTM data persists until explicitly deleted, while STM checkpoints follow the configured expiry policy.
- OpenAI calls occur during indexing, generation, fallback embedding, and metric judging. A successful local run can incur usage and latency.
- Shared Redis clients use two-second connection timeouts, five-second command timeouts, health checks, and two exponential-backoff retries. `make doctor` reports the configured STM expiry and uses the same client factory.
- The Flex RAG, memory, and Workbench application initialization is explicit; imports should not create clients, indexes, embedders, or fetch content. Other demos still perform setup as part of their script lifecycle, so run them as standalone programs.
- Long-running or externally backed operations remain bounded: Flex RAG caps rewrites/model retries/graph recursion, and evaluation batch collection has a one-check `--partial` path.

## Verification expectations

Run `make verify` for Ruff, the default fast and Redis integration tests, and bytecode compilation of the application directories. The detailed layers and direct commands are in [Testing and verification](testing.md). Redis-dependent tests skip with an explicit reason when local capabilities are unavailable; CI repeats the required gate on Python 3.13 with Redis 8. Website and OpenAI smoke tests are separate, explicitly opt-in checks exposed through the manually dispatched live-integrations workflow. Verify changes by running the smallest affected script with a disposable Redis instance and configured API access. For query changes, inspect dimensions, dtype, index algorithm, namespace/prefix, distance metric, and cleanup. For memory changes, verify STM TTL/refresh behavior, LTM persistence, and user-scoped deletion. For evaluation changes, preserve the recorded chunking configuration and retrieve-once scoring contract.
