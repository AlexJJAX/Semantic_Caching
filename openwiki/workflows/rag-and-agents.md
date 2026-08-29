# RAG and agent workflows

## Role-based RAG

[`RAG/User_role_based_rag.py`](../../RAG/User_role_based_rag.py) models users in RedisJSON, assigns roles, ingests PDF chunks, and stores an `allowed_roles` tag alongside embeddings. Retrieval adds a Redis tag filter for the requesting user's roles. Unauthorized vector hits are removed before the application or LLM sees them; a no-hit path returns a permission-safe response without making an OpenAI call.

The example maps document filename characteristics to role sets and uses user IDs such as `alice`, `larry`, and `tyler`. Ingestion is content-addressed and replaces only the document's prior chunks; retrieval applies a maximum distance as well as the role filter. Results preserve source/page metadata, delimit passages as untrusted evidence, and append citations. Treat this as a security pattern demonstration, not a complete identity provider or policy engine.

## Self-correcting Flex RAG

[`agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py`](../../agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py) compiles a `MessagesState` graph:

```text
START → agent → retrieve → grade_documents → generate
             ↘ END             ↘ rewrite → agent
```

The agent uses an OpenAI tool-calling model and a Redis retriever over hard-coded Lilian Weng sources. On first use, the script can fetch, split, embed, and index those URLs; an existing `rag-redis` index is reused. A structured relevance grader selects generation versus query rewriting. The source README and implementation should be read together because importing the module performs setup and connectivity work.

The rewrite path is explicitly bounded (two rewrites by default, with model and graph recursion limits); exhausted retries return an insufficient-context response rather than looping. Initialization is provided by an application factory, so imports are safe for routing tests. Retrieved source URLs are carried into untrusted, delimited context and a deterministic source list.

## Dual memory travel assistant

[`agentic/Memory/agentic_memory.py`](../../agentic/Memory/agentic_memory.py) combines:

- **Short-term memory (STM):** LangGraph state persisted with a Redis checkpoint saver, scoped by `thread_id`.
- **Long-term memory (LTM):** RedisVL JSON vector records, scoped by `user_id`, split into episodic preferences and semantic travel facts.

The agent can call storage/retrieval tools. Storage embeds and performs similarity deduplication; retrieval applies user/type filters and similarity thresholds. After six or more messages, the graph summarizes the conversation and removes older messages.

The memory application now reopens both indexes with `overwrite=False` and uses an explicit factory. STM checkpoint keys receive configurable sliding expiry (24 hours by default); LTM records have no TTL and retain provenance until explicit deletion. LTM retrieval and deduplication are scoped by `user_id`, while deletion verifies both memory ID and owner. The shared Redis pool closes on exit without deleting retained state.

## Multi-session history

[`llm_message_history/Multiple_sessions.py`](../../llm_message_history/Multiple_sessions.py) creates one RedisVL `SemanticMessageHistory` and partitions conversations with `session_tag` values for three personas. It loads each session, sends the same prompt with persona context to OpenAI, stores the response, prints a session, and cleans only its own namespaced history/index in a failure-safe exit path. Session isolation depends on always supplying the intended tag.
