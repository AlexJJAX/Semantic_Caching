# Bounded Agentic RAG with LangGraph, local Redis & GPT-5.6 Luna

![Tests](https://img.shields.io/badge/focused%20tests-4%20passing-2E7D32)
![Model](https://img.shields.io/badge/model-gpt--5.6--luna-412991)
![Redis](https://img.shields.io/badge/Redis-8%20Search-DC382D)
![Architecture](https://img.shields.io/badge/architecture-bounded%20agentic%20RAG-00796B)
![License](https://img.shields.io/badge/license-MIT-455A64)

A local instance Redis-backed question-answering agent that can decide to retrieve, grade the evidence it
finds, rewrite an ineffective query, and stop cleanly when the corpus cannot support an
answer.

The example uses LangGraph to make every branch explicit. Redis persists the vector index,
OpenAI embeddings power semantic retrieval, and GPT-5.6 Luna performs tool selection,
relevance grading, query rewriting, and grounded generation.

This is a generic, demonstrational working primitive intended to showcase Redis retrieval inside
a bounded agentic RAG loop. It is not intended or suitable for production use: the fixed corpus,
prompts, thresholds, retry limits, persistence choices, and model decisions are explanatory
defaults rather than service guarantees or a complete agent platform.

## Architecture Overview

| Component                       | Responsibility                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `FlexRagState`                  | Extends LangGraph message state with the relevance decision and rewrite count                    |
| `load_or_create_vectorstore()`  | Reuses the existing Redis Search index or fetches, chunks, embeds, and indexes the source corpus |
| Redis retriever tool            | Returns up to four passages that meet the configured similarity threshold                        |
| Agent node                      | Decides whether the current question requires the retrieval tool                                 |
| Grader node                     | Produces a structured `yes` / `no` assessment of the retrieved context                           |
| Rewrite node                    | Adds an improved query as a real `HumanMessage`, then returns control to the agent               |
| Generate node                   | Answers the original question from retrieved context and appends deduplicated source URLs        |
| Fallback node                   | Returns an explicit insufficient-context response when the rewrite budget is exhausted           |
| `create_flex_rag_application()` | Performs external initialization explicitly and returns the graph plus its owned Redis resource  |
| CLI entry point                 | Streams graph updates and closes the Redis connection pool in a `finally` block                  |

All executable behavior lives in
[`Langgraph_redis_agentic_flex_rag.py`](./Langgraph_redis_agentic_flex_rag.py). Importing the
module does not create clients, fetch web pages, build indexes, or call a model.

## Why this is agentic RAG

This is not a fixed `retrieve → generate` chain. The model can choose whether retrieval is
necessary, inspect the result through a separate relevance grader, and alter the search query
when the first retrieval is weak.

That autonomy is deliberately bounded. The default budget allows two rewrites, each
model-backed graph node has at most two attempts, and the CLI derives a finite recursion limit
from the rewrite budget. An irrelevant corpus therefore terminates with a clear fallback
instead of creating an unbounded self-correction loop.

## What it demonstrates

- Conditional LangGraph routing around a model-bound Redis retrieval tool.
- Persistent vector search with Redis Search and `langchain-redis`.
- Retrieval-time filtering with a `0.7` similarity threshold and `k=4`.
- Structured relevance grading before generation.
- Query correction that changes the next retrieval input, not just prompt-local text.
- Bounded model retries, rewrites, and graph recursion.
- Source metadata propagation from ingestion through retrieval to the final answer.
- Prompt-injection boundaries around retrieved web content.
- Explicit application construction and deterministic resource cleanup.

## Key Design Decisions

- **Filter before grading** — the Redis-backed retriever removes weak vector matches before any
  context reaches the grader. The grader then provides a second, semantic relevance gate.
- **Preserve the user's intent** — rewrites become new `HumanMessage` objects for subsequent
  searches, while generation still answers the original question.
- **Fail closed on missing evidence** — when the rewrite budget is exhausted, the graph states
  that it cannot answer confidently instead of generating from unrelated passages.
- **Treat retrieved text as data** — each passage is wrapped in
  `<untrusted_retrieved_passage>` delimiters. Both grader and generator prompts instruct the
  model to ignore instructions found inside that content.
- **Make provenance visible** — every formatted passage carries its source URL. Generation asks
  for inline citations, and the application also appends a deterministic, deduplicated
  `Sources` section.
- **Reuse expensive work** — the namespaced Redis index persists between runs. Source pages are
  fetched and embedded only when that index is absent.
- **Keep lifecycle ownership explicit** — the application disconnects its Redis connection pool
  on exit without deleting the persisted index or unrelated Redis data.

## Grounding and limitations

Grounding, bounded retries, source metadata, and index reuse make this Redis primitive inspectable;
they do not make model output trustworthy by default or make the application production-ready.

1. The strict context-only generation contract applies after retrieval. Because this is a
   tool-selection example, the agent may also decide to answer directly and end the graph
   without entering the grounded generation branch.
2. The corpus contains three fixed Lilian Weng articles. It is intentionally narrow and is not
   a general web-search system.
3. Reusing the Redis index makes repeat runs fast, but source changes are not refreshed
   automatically. Rebuilding requires removing this example's owned index and document keys.
4. Relevance grading is model-based and therefore probabilistic. The Redis similarity threshold
   reduces weak candidates but does not prove that a passage answers the question.
5. Prompt delimiters and instructions reduce indirect prompt-injection risk; they are an
   application boundary, not a substitute for content validation and model-output controls in a
   production system.

## Redis index and source corpus

| Setting             | Default                                           |
| ------------------- | ------------------------------------------------- |
| Search index        | `{REDIS_NAMESPACE}:idx:flex-rag`                  |
| Document key prefix | `{REDIS_NAMESPACE}:flex-rag:document`             |
| Index algorithm     | `FLAT`                                            |
| Embedding model     | `text-embedding-3-small`                          |
| Chunking            | 500 tokens with 100-token overlap                 |
| Retrieval           | Top 4 passages with similarity score ≥ `0.7`      |
| Persistence         | Reuse the index when `FT.INFO` confirms it exists |

The initial index is built from:

- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/)

With the default namespace, the concrete index name is `portfolio:idx:flex-rag`. A custom
`REDIS_NAMESPACE` keeps independent runs isolated.

## Simplified Request flow

```text
1. User question enters the LangGraph message state
2. Agent decides whether to call retrieve_blog_posts
   ├─ no tool call → return the agent response and end
   └─ tool call
      3. Redis returns up to four threshold-qualified passages
      4. Grader evaluates the passages as untrusted context
         ├─ relevant → answer the original question → append sources → end
         └─ irrelevant
            ├─ rewrite budget available → append rewritten HumanMessage → agent
            └─ budget exhausted → return insufficient-context response → end
```

For detailed sequence diagram depicting the end-to-end Flex RAG request flow, refer to the [flex_rag_sequence_diagram.md](agentic/Flex_rag/flex_rag_sequence_diagram.md) and preview it in https://mermaid.live/

## Path Summary

| Path                    | When                                            | Terminal Node            |
| ----------------------- | ----------------------------------------------- | ------------------------ |
| **Direct answer**       | Agent decides retrieval is unnecessary          | `agent → END`            |
| **Grounded generation** | Retrieved passages pass relevance grading       | `generate → END`         |
| **Rewrite → retry**     | Passages fail grading, rewrite budget remains   | `rewrite → agent` (loop) |
| **Fallback**            | Passages fail grading, rewrite budget exhausted | `not_found → END`        |

## Key Participants

| Participant | Implementation                                                                                                       |
| ----------- | -------------------------------------------------------------------------------------------------------------------- |
| Agent       | [`call_agent`](agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py#L251-L254) — `model_with_tools.invoke()`         |
| Retriever   | [`ToolNode`](agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py#L304) — wraps `retrieve_blog_posts`                |
| Grader      | [`grade_documents`](agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py#L256-L263) — structured `GradeScore` output |
| Rewriter    | [`rewrite_question`](agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py#L265-L273) — appends new `HumanMessage`    |
| Generator   | [`generate_answer`](agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py#L275-L284) — context-only + source URLs     |
| Fallback    | [`no_answer_found`](agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py#L286-L296) — explicit insufficient-context  |
| Router      | [`route_after_grading`](agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py#L134-L144) — 3-way conditional edge     |

## Run it

Prerequisites:

- Python 3.13 or later.
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.
- A [local Redis 8](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/) instance with Search and JSON commands available.
- An OpenAI API key. The first run also needs network access to fetch the three source pages.

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

# Run the default question.
uv run python agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py
```

Ask a different question or change the rewrite budget:

```bash
uv run python agentic/Flex_rag/Langgraph_redis_agentic_flex_rag.py \
  --question "How do agents use short-term and long-term memory?" \
  --max-rewrites 2
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

## Expected output

The CLI prints whether it is building or reusing the index, then streams one update per graph
node. The following is representative and abbreviated because message IDs and model metadata
vary by run:

```text
Connecting to Redis at redis://localhost:6379/0
Reusing Redis index 'portfolio:idx:flex-rag'.
---CALL AGENT---
{'agent': {'messages': [AIMessage(... tool_calls=[...])]}}
---CHECK RELEVANCE---
---DOCUMENTS RELEVANT---
{'grade_documents': {'documents_relevant': True}}
---GENERATE---
{'generate': {'messages': [AIMessage(content='...\n\nSources:\n- https://...')]}}
```

On a cold run, the reuse line is replaced by an index-build message followed by the number of
chunks indexed. If the retrieved passages remain irrelevant after the configured rewrites, the
final update contains:

```text
I couldn't find relevant context after the allowed query rewrites, so I can't answer this confidently.
```

Other useful prompts to try:

- `What memory types are used by LLM agents?`
- `How does chain-of-thought differ from tree-of-thought reasoning?`
- `What attack methods are described for language models?`
- `What does the corpus say about a topic it does not cover?`

## Test it

The focused routing and safety tests remain fast and do not contact Redis, the source websites,
or OpenAI:

```bash
uv run python -m unittest \
  tests.test_phase2.FlexRoutingTests \
  tests.test_phase2_requirements.FlexRagSafetyTests -v
```

The opt-in website integration executes the example's real loader, including its request timeout,
against all three configured source URLs and validates parsed content and source metadata:

```bash
RUN_LIVE_WEB_TESTS=1 \
  uv run python -m unittest tests.test_live_integrations.LiveSourceWebsiteTests -v
```

Run the repository-wide quality gate directly when Redis is available:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench
```

`make verify` is the optional convenience alias for these commands.
See the repository [test strategy](../../TESTING.md) for live-test gates and the complete coverage
matrix.

## License

This project is available under the repository's [MIT License](../../LICENSE).
