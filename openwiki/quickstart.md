# Semantic Caching & AI Portfolio

## What this repository is

This repository is a collection of runnable Python demonstrations for Redis-backed AI application patterns: semantic caching, vector search, RAG, role-aware retrieval, LangGraph agents, persistent memory, multi-session chat history, and Ragas evaluation. The Redis AI Workbench provides a local web view across four of those patterns, but the repository remains a portfolio of separate experiments rather than one deployable service.

The root [`README.md`](../README.md) is the source-level overview. This wiki adds navigation, explains how the examples relate, and records operational caveats discovered in the implementations.

## Start here

1. Install Python `3.13+` and `uv` (the authoritative version constraint is [`pyproject.toml`](../pyproject.toml)).
2. Run `make setup` from the repository root; this performs a locked `uv sync`.
3. Start a Redis deployment with Search, JSON, and vector support. For the repository's local workflow, use `make redis-start`.
4. Copy `.env.example` to `.env` and configure `OPENAI_API_KEY` for OpenAI-backed examples. `REDIS_URL` takes precedence over the individual Redis settings; the shared configuration also supports `REDIS_NAMESPACE`, cache thresholds, and TTL.
5. Run `make doctor` to validate Python, resources, configuration, Redis connectivity, `FT.CREATE`, and `JSON.SET` before a demo.
6. Open the Workbench with `make workbench` and visit `http://127.0.0.1:8123`; use `WORKBENCH_MODEL_MODE=demo` for the repeatable local simulator.
7. Execute scripts from the repository root because their relative paths expect `resources/` to be at the working-directory root.

Example:

```bash
make setup
make redis-start
make doctor
uv run python vector_search/1_redispy/Redispy.py
```

Do not read or commit live secret files. The repository contains environment-related files; use placeholders and rotate any exposed credentials before sharing the checkout. The checked-in environment file is sensitive and is not a valid documentation template.

## Major sections

- [Architecture overview](architecture/overview.md) — how the independent demos fit together and which Redis/LangChain layers they exercise.
- [RAG and agent workflows](workflows/rag-and-agents.md) — RBAC RAG, self-correcting retrieval, dual memory, and session history.
- [Search and evaluation](search-and-evaluation.md) — the three vector-search implementations and the Ragas batch/evaluation pipeline.
- [Operations](operations.md) — prerequisites, commands, configuration, cleanup behavior, and risks; includes the Workbench runbook.
- [Testing and verification](testing.md) — fast, Redis integration, and explicitly opt-in live test layers with commands and change guidance.
- [Source map](source-map.md) — where to continue reading for each capability, including semantic caching and the Workbench.

## Repository domains

| Domain | Entry point | Primary input/output |
|---|---|---|
| Semantic cache | `semantic_cache/semantic_cache_demo.py` | RedisJSON exact/semantic cache entries, TTL/invalidation, calibration and cost/latency metrics |
| Redis AI Workbench | `workbench/server.py` via `make workbench` | Local HTTP/SSE UI over cache, STM/LTM, RBAC RAG, and retrieval evaluation |
| Native vector search | `vector_search/1_redispy/Redispy.py` | `resources/movies.json`; raw RediSearch queries and printed results |
| RedisVL search | `vector_search/2_redisvl/Redisvl.py` | Same movie data; schema/query-object demonstrations |
| Multi-vector search | `vector_search/3_multivector_search/Multivector_search.py` | Same movie data; three embeddings and weighted ranking |
| RBAC RAG | `RAG/User_role_based_rag.py` | Corporate PDFs; RedisJSON users and permission-filtered context |
| Flex RAG | `agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py` | Hard-coded web sources; tool retrieval, grading, rewriting, generation |
| Agent memory | `agentic/Memory/agentic_memory.py` | User/thread conversation state and RedisVL long-term memories |
| Session history | `llm_message_history/Multiple_sessions.py` | Three `session_tag`-partitioned conversations |
| Evaluation | `evaluation/generate_testset.py`, `evaluation/06_ragas_evaluation.py` | Nike 10-K chunks, synthetic CSV, Ragas metrics CSV |

## Change-oriented guidance

- For semantic-cache behavior, start with [`semantic_cache/README.md`](../semantic_cache/README.md) and `src/redis_ai_portfolio/semantic_cache.py`; preserve exact-first lookup, partition filters, false-hit guards, bypass policy, TTL, and scoped invalidation when changing it.
- For the Workbench, start with [`workbench/README.md`](../workbench/README.md), `src/redis_ai_portfolio/workbench.py`, and `workbench/server.py`; keep live/demo mode explicit, SSE recorder behavior bounded, and Redis inspection sanitized.
- For Redis query behavior, start with the relevant script and its local README; verify index algorithm, vector dimensions, dtype, namespace/prefix, and cleanup before changing it.
- Shared configuration and Redis naming live in [`src/redis_ai_portfolio/config.py`](../src/redis_ai_portfolio/config.py), while bounded client construction lives in [`src/redis_ai_portfolio/redis.py`](../src/redis_ai_portfolio/redis.py); use `PortfolioSettings.redis_name()` for new keys and indexes so demos can coexist.
- For agent changes, inspect graph state and routing functions before prompts. `thread_id` scopes short-term state, which expires according to `STM_TTL_MINUTES`/`STM_REFRESH_TTL_ON_READ`; `user_id` scopes long-term memory.
- For evaluation changes, keep testset chunking aligned between generation and scoring and watch the version-sensitive Ragas imports.
- Configuration and offline behavior are covered by [`tests/test_config.py`](../tests/test_config.py) and [`tests/test_phase2_requirements.py`](../tests/test_phase2_requirements.py); message-history behavior and session isolation by [`tests/test_message_history.py`](../tests/test_message_history.py); all three vector-search demos by [`tests/test_vector_search_redis.py`](../tests/test_vector_search_redis.py); Redis lifecycle, retention, migration, and namespace isolation by [`tests/test_redis_integration.py`](../tests/test_redis_integration.py). See [Testing and verification](testing.md) for the layered commands and live-test boundaries; run `make verify` before relying on a change.

## Backlog

- Add a common benchmark harness for comparing the three vector-search demos.
- Reconcile README Python claims (`3.10+`) with the project constraint (`>=3.13`).
- Document a supported Redis deployment and reproducible local startup command.
