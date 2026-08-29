# Source map

| Area | Read first | Follow-up |
|---|---|---|
| Portfolio setup | [`README.md`](../README.md) | [`pyproject.toml`](../pyproject.toml), [`Makefile`](../Makefile) |
| Shared configuration | [`src/redis_ai_portfolio/config.py`](../src/redis_ai_portfolio/config.py) | [`src/redis_ai_portfolio/doctor.py`](../src/redis_ai_portfolio/doctor.py), [`tests/test_config.py`](../tests/test_config.py) |
| Semantic cache | [`semantic_cache/README.md`](../semantic_cache/README.md) | [`src/redis_ai_portfolio/semantic_cache.py`](../src/redis_ai_portfolio/semantic_cache.py), [`semantic_cache/semantic_cache_demo.py`](../semantic_cache/semantic_cache_demo.py), [`tests/test_semantic_cache.py`](../tests/test_semantic_cache.py) |
| Redis AI Workbench | [`workbench/README.md`](../workbench/README.md) | [`src/redis_ai_portfolio/workbench.py`](../src/redis_ai_portfolio/workbench.py), [`workbench/server.py`](../workbench/server.py), [`tests/test_workbench.py`](../tests/test_workbench.py) |
| RBAC RAG | [`RAG/README.md`](../RAG/README.md) | [`RAG/User_role_based_rag.py`](../RAG/User_role_based_rag.py) |
| Self-correcting RAG | [`agentic/Flex_rag/README.md`](../agentic/Flex_rag/README.md) | [`Langgraph_redis_agentic_flex_rag.py`](../agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py) |
| Dual memory | [`agentic/Memory/README.md`](../agentic/Memory/README.md) | [`agentic_memory.py`](../agentic/Memory/agentic_memory.py) |
| Session history | [`llm_message_history/README.md`](../llm_message_history/README.md) | [`Multiple_sessions.py`](../llm_message_history/Multiple_sessions.py) |
| Native Redis search | [`vector_search/1_redispy/README.md`](../vector_search/1_redispy/README.md) | [`Redispy.py`](../vector_search/1_redispy/Redispy.py) |
| RedisVL search | [`vector_search/2_redisvl/README.md`](../vector_search/2_redisvl/README.md) | [`Redisvl.py`](../vector_search/2_redisvl/Redisvl.py) |
| Multi-vector search | [`vector_search/3_multivector_search/README.md`](../vector_search/3_multivector_search/README.md) | [`Multivector_search.py`](../vector_search/3_multivector_search/Multivector_search.py) |
| Testset generation | [`evaluation/README.md`](../evaluation/README.md) | [`generate_testset.py`](../evaluation/generate_testset.py) |
| Ragas scoring | [`evaluation/README.md`](../evaluation/README.md) | [`06_ragas_evaluation.py`](../evaluation/06_ragas_evaluation.py) |
| Shared data | `resources/movies.json` and PDFs | Consumers listed in [`quickstart.md`](quickstart.md) |
| Verification | [`TESTING.md`](../TESTING.md) | [`Makefile`](../Makefile), [`tests/test_message_history.py`](../tests/test_message_history.py), [`tests/test_vector_search_redis.py`](../tests/test_vector_search_redis.py), [`tests/test_live_integrations.py`](../tests/test_live_integrations.py), [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), [`.github/workflows/live-integrations.yml`](../.github/workflows/live-integrations.yml) |

## Git context

Recent project commits add the semantic-cache engine and the Redis AI Workbench on top of the shared typed configuration, Redis lifecycle targets, and namespace-qualified cleanup documented here. The Workbench and cache tests are the first checks to inspect when changing those areas; generated wiki pages remain confined to `openwiki/`.
