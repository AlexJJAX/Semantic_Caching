# Architecture overview

## Shape of the repository

The project is a shared `uv` dependency environment containing independent scripts. Redis is the common state and retrieval layer; LangChain/LangGraph provide orchestration in the RAG and agent examples; OpenAI and HuggingFace provide generation and embeddings; Ragas evaluates one Redis-backed RAG path.

```text
resources/ (PDFs, movie JSON)
        │
        ├── Redis vector demos ──> RediSearch indexes ──> ranked results
        ├── RBAC RAG ────────────> role-filtered chunks ──> permitted LLM context
        ├── Flex RAG ────────────> tool retrieval ──> grade/rewrite/generate graph
        ├── Memory agent ────────> Redis checkpoints + vector memories
        ├── Evaluation ─────────> Redis retriever + Ragas metrics
        ├── Semantic cache ─────> exact JSON lookup + filtered vector lookup + TTL
        └── Workbench ──────────> localhost HTTP/SSE portfolio and Redis inspector
```

## Shared infrastructure

`pyproject.toml` supplies Redis clients, `langchain-redis`, LangGraph checkpointing, OpenAI/HuggingFace integrations, Ragas, PDF parsing, and RedisVL. The shared [`src/redis_ai_portfolio/config.py`](../../src/redis_ai_portfolio/config.py) layer loads `.env` without overriding process variables, validates Redis/OpenAI/cache settings, builds or accepts a Redis URL, redacts credentials for display, and creates lowercase namespace-qualified names for keys and indexes. Most scripts use this settings object and assume execution from the repository root.

Redis Stack capabilities matter: vector search requires RediSearch, and the RBAC implementation uses RedisJSON for user objects. The [`doctor.py`](../../src/redis_ai_portfolio/doctor.py) command checks these capabilities and required resources before a run. A plain Redis server is not sufficient for the full portfolio.

## Architectural boundaries

### Retrieval implementations

- `vector_search/1_redispy` uses raw Redis commands and a HASH/HNSW index.
- `vector_search/2_redisvl` uses RedisVL schemas and query objects with a FLAT index and embedding cache.
- `vector_search/3_multivector_search` stores three model-specific vector fields and blends weighted distances in one query.

These are demonstrations, not a controlled benchmark: algorithms, models, dtypes, and scoring differ.

### Semantic cache

`src/redis_ai_portfolio/semantic_cache.py` implements a cache-aside boundary: canonicalized prompts are checked by exact RedisJSON key before embedding; misses perform partition-filtered cosine vector search, apply conservative false-hit guards, then call the model and write a JSON entry with TTL and invalidation tags. Tenant, task, model, prompt version, and permission scope are part of the partition, preventing cross-context reuse. The module also exposes invalidation, feedback, calibration, and cost/latency metrics.

### Redis AI Workbench

`src/redis_ai_portfolio/workbench.py` composes the cache, memory, RBAC retrieval, and evaluation demonstrations behind a run recorder. `workbench/server.py` is a small localhost-only HTTP server that serves the static UI, JSON routes, and replayable Server-Sent Events. Live mode uses OpenAI Responses/Embeddings APIs; explicit demo mode uses deterministic local embeddings and simulated responses. The Redis inspector exposes metadata only, not stored values.

### RAG and agents

- `RAG/User_role_based_rag.py` applies authorization in the Redis query before context reaches the model.
- `agentic/Flex_rag` compiles a LangGraph loop that decides whether to retrieve, grades relevance, rewrites failed queries, and generates an answer.
- `agentic/Memory` separates checkpointed short-term conversation state from user-scoped vector memories.
- `llm_message_history` demonstrates simpler session partitioning with RedisVL `SemanticMessageHistory`.

### Evaluation

`evaluation/generate_testset.py` creates synthetic questions using an asynchronous OpenAI Batch embedding workflow. `evaluation/06_ragas_evaluation.py` rebuilds the Redis RAG index, preserves retrieved contexts, and scores faithfulness, answer relevancy, context recall, and context precision.

## Why the code is organized this way

The initial commit presents progressively higher-level abstractions: raw `redis-py`, RedisVL, multi-vector retrieval, then application-level RAG and agents. Phase 2 keeps those learning boundaries while extracting shared Redis lifecycle behavior and making application factories explicit, so imports can remain testable and each example can own only its namespace. Lifecycle details are not globally uniform: STM checkpoints expire by policy, while provenance-bearing LTM and selected RAG indexes persist; semantic-cache entries use absolute TTLs and scoped invalidation.
