# Authorization-Filtered RAG with Redis & GPT-5.6 Luna

![Tests](https://img.shields.io/badge/focused%20tests-2%20passing-2E7D32)
![Integration](https://img.shields.io/badge/Redis%20integration-1%20passing-00796B)
![Model](https://img.shields.io/badge/model-gpt--5.6--luna-412991)
![Redis](https://img.shields.io/badge/Redis-8%20Search%20%7C%20JSON-DC382D)
![Architecture](https://img.shields.io/badge/architecture-RBAC%20RAG-00796B)
![License](https://img.shields.io/badge/license-MIT-455A64)

A Redis-backed retrieval-augmented generation workflow that enforces document permissions
inside vector search. User roles are read from RedisJSON, document chunks carry indexed
`allowed_roles` tags, and Redis removes unauthorized candidates before retrieved text can reach
the application or GPT-5.6 Luna.

The example also demonstrates content-addressed PDF ingestion, bounded similarity retrieval,
source/page provenance, explicit prompt-injection boundaries, per-user in-process session
history, and persistent Redis state with scoped resource ownership.

This is a generic, demonstrational working primitive intended to showcase Redis-backed,
authorization-filtered RAG. It is not intended or suitable for production use: its fixed users,
role rules, documents, thresholds, prompts, persistence, and cleanup make the retrieval boundary
observable but do not constitute an identity, policy, security, or data-governance system.

## Architecture Overview

| Component                                                      | Responsibility                                                                                                  |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `User`                                                         | Stores and manages a user's normalized role set in RedisJSON                                                    |
| `KnowledgeBase`                                                | Chunks PDFs, assigns access roles, creates OpenAI embeddings, and persists source-aware RedisJSON records       |
| Redis `VectorRangeQuery`                                       | Applies the role filter and cosine-distance threshold together before returning ranked passages                 |
| `user_query()`                                                 | Loads the requesting user's current roles and performs a permission-filtered search without generation          |
| `RAGChatManager`                                               | Owns per-user message histories, prepares bounded evidence, calls GPT-5.6 Luna, and stores successful exchanges |
| Context formatter                                              | Labels each retrieved passage with source, page, and chunk metadata and marks it as untrusted data              |
| Source formatter                                               | Appends a deterministic source list derived from exactly the passages returned by Redis                         |
| [`rbac_rag_sequence_diagram.md`](rbac_rag_sequence_diagram.md) | Traces identity lookup, ingestion, authorization-filtered retrieval, generation paths, persistence, and cleanup |

All executable behavior lives in
[`User_role_based_rag.py`](./User_role_based_rag.py). Importing the module loads shared
configuration and class definitions but does not connect to Redis, ingest files, or call OpenAI.

### End-to-end RBAC RAG sequence diagram

For the complete interaction among the caller, user store, knowledge base, Redis Search,
message history, embeddings API, and GPT-5.6 Luna, see the
[end-to-end RBAC RAG sequence diagram](rbac_rag_sequence_diagram.md).

## Authorization contract

```text
user_id
  ↓
RedisJSON user lookup
  ├─ missing user or empty roles ───────────────→ reject before retrieval
  ↓
Current role set
  ↓
Embed query
  ↓
Redis VectorRangeQuery
  ├─ allowed_roles TAG filter
  ├─ cosine distance ≤ 0.3
  └─ top 5 results
       ├─ no qualified passages ────────────────→ permission-safe response; no LLM call
       └─ authorized passages only
              ↓
         untrusted-evidence prompt boundary
              ↓
         GPT-5.6 Luna → cited answer → per-user history
```

The central control is retrieval-time filtering:

```python
Tag("allowed_roles") == user_roles
```

That expression is passed to the same Redis vector query that enforces distance and result
limits. The system does not retrieve unrestricted passages and ask the model to ignore those it
should not see.

## What it demonstrates

- RedisJSON users with validated, deduplicated role values.
- RedisJSON document chunks with multi-value `allowed_roles` TAG indexing.
- Authorization applied inside Redis vector retrieval.
- Fail-closed handling for missing users, users without roles, and empty authorized result sets.
- `FLAT`, `FLOAT32`, cosine vector search with a maximum distance threshold.
- Content-addressed document IDs and idempotent replacement of one document's chunks.
- Durable index reuse plus non-destructive citation-schema upgrades.
- Source filename, page number, and chunk identity propagated into generated evidence.
- Explicit untrusted-passage delimiters and a context-only generation policy.
- Deterministic source lists appended after model generation.
- Per-user RedisVL message history kept separate from the authorization decision.
- Bounded Redis connection and command timeouts through the shared client factory.
- Explicit OpenAI and Redis client closure while user, document, and session data remain durable.

## Key Design Decisions

- **Authorize during retrieval** — Redis combines semantic distance with the role TAG filter, so
  an unauthorized vector match never becomes application context.
- **Reload roles for every request** — `RAGChatManager.answer()` reads the user record again
  before searching. Role additions and removals affect new retrievals without restarting the
  application.
- **Keep identity and evidence separate** — roles live on user JSON documents; permissions live
  on every knowledge chunk. Session messages are not trusted as an authorization source.
- **Stop before generation when evidence is unavailable** — an empty authorized result returns
  a fixed, permission-safe response without spending a model call.
- **Use distance and permissions together** — access eligibility alone is insufficient. A
  passage must also be semantically close enough to the question and fall within the top-five
  result budget.
- **Treat retrieved content as data** — every passage is wrapped in
  `<untrusted_retrieved_passage>` delimiters, and the system policy tells the model not to follow
  instructions embedded in retrieved text.
- **Make provenance visible** — numbered evidence blocks include filename, one-based PDF page,
  and chunk ID. The final answer receives a source list built from those same Redis results.
- **Replace only owned document chunks** — ingestion derives a document ID from the source bytes,
  removes only matching keys, and leaves other documents and indexes untouched.
- **Preserve durable state** — the document index is opened with `overwrite=False`. Shutdown
  closes clients but intentionally retains users, knowledge chunks, and message histories.

## Redis data model

### Users

Each user is stored by direct key lookup:

```text
{REDIS_NAMESPACE}:rbac-rag:user:{user_id}
```

```json
{
  "user_id": "alice",
  "roles": ["executive", "finance"]
}
```

Supported demonstration roles are `finance`, `manager`, `executive`, `hr`, `sales`, and
`product`. `User.save()` normalizes supplied values through the `UserRoles` enum, removes
duplicates, sorts them, and rejects unknown role names.

### Knowledge chunks

| Setting             | Default                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| Search index        | `{REDIS_NAMESPACE}:idx:rbac-rag:documents`                                |
| Document key prefix | `{REDIS_NAMESPACE}:rbac-rag:document:`                                    |
| Document key        | `{prefix}{16-character-content-hash}:chunk_{n}`                           |
| Storage type        | RedisJSON                                                                 |
| Chunking            | 512 characters with 100-character overlap                                 |
| Embedding model     | `text-embedding-3-small`                                                  |
| Vector index        | `FLAT`, `FLOAT32`, cosine distance; dimensions reported by the vectorizer |
| Retrieval           | Up to 5 passages with cosine distance ≤ `0.3`                             |
| Permission match    | Any indexed `allowed_roles` value matching a current user role            |
| Persistence         | Index and documents are retained across runs                              |

Each chunk contains:

```text
doc_id          Content-derived document identifier
chunk_id        Stable position within that document
allowed_roles   Redis TAG array used by the authorization filter
content         Retrieved passage text
source          Original source path
page            One-based PDF page number
embedding       OpenAI vector used for cosine search
```

`ensure_citation_schema()` adds `source` and `page` fields to an older compatible index through
idempotent `FT.ALTER` calls. It does not recreate the index or delete existing documents.

### Per-user message history

`RAGChatManager` creates one RedisVL `MessageHistory` index per user when that user first asks a
question:

```text
{REDIS_NAMESPACE}:rbac-rag:session:{user_id}
```

The index and its Hash key prefix share that name. RedisVL also generates a default
`session_tag` for each new `MessageHistory` instance. Within one process, the same instance is
reused from `RAGChatManager.sessions`, and its five most recent messages are included after the
system policy and before the next evidence-bearing user message. RedisVL's `llm` role is
converted to OpenAI's `assistant` role before generation.

History keys have no application-configured TTL and remain after shutdown. A new process creates
a new default `session_tag`, however, so these records are durable but are not automatically
restored into the next run's chat context.

## Demonstration role mapping

When `allowed_roles` is omitted, `_determine_roles()` assigns roles from filename patterns:

| Filename signal                           | Assigned roles         | Included example            |
| ----------------------------------------- | ---------------------- | --------------------------- |
| `10k`, `financial`, `earnings`, `revenue` | `finance`, `executive` | Apple 10-K                  |
| `brochure`, `spec`, `product`, `manual`   | `product`, `sales`     | Chevrolet Colorado brochure |
| `hr`, `handbook`, `policy`, `employee`    | `hr`, `manager`        | Not included                |
| `sales`, `pricing`, `customer`            | `sales`, `manager`     | Not included                |
| No pattern match                          | `executive`            | Fallback only               |

This is intentionally visible business logic for the example. Production ingestion should use
authoritative document policy metadata, not infer access rights from filenames.

## Grounding, citations, and prompt boundaries

Authorized Redis results are formatted as numbered blocks:

```text
[SOURCE 1] 2022-chevrolet-commercial-colorado-ebrochure.pdf, page 4, chunk chunk_12
<untrusted_retrieved_passage>
...
</untrusted_retrieved_passage>
```

The model receives:

1. The configured assistant prompt plus `RAG_SECURITY_POLICY`.
2. Recent history for the same `user_id`, when present.
3. A user message containing the original question and retrieved evidence inside a separate
   `<retrieved_evidence>` boundary.

The policy requires context-only answers, numbered citations, refusal when evidence is
insufficient, and rejection of instructions found in retrieved passages. After generation, the
application appends a `Sources` section from exactly the returned Redis documents, independent
of whether the model formatted its inline citations perfectly.

## Demonstrated access paths

The script executes these checks in order:

| Request                                 | Stored roles           | Relevant document roles     | Expected path                                                      |
| --------------------------------------- | ---------------------- | --------------------------- | ------------------------------------------------------------------ |
| Unknown `tyler` before creation         | None                   | `product`, `sales`          | Reject user lookup before vector search                            |
| `tyler` asks about the Chevrolet        | `sales`                | `product`, `sales`          | Return authorized brochure chunks                                  |
| `alice` asks about the Chevrolet        | `finance`, `executive` | `product`, `sales`          | Return no brochure chunks                                          |
| `alice` asks about Apple revenue        | `finance`, `executive` | `finance`, `executive`      | Return authorized 10-K chunks                                      |
| Chat request with no qualified evidence | Current user roles     | No role-and-distance match  | Store fixed no-document response; skip GPT-5.6 Luna                |
| Chat request with qualified evidence    | Current user roles     | Matching roles and distance | Generate a grounded answer, append sources, and store the exchange |

The sequence diagram shows these branches in system-interaction order:
[rbac_rag_sequence_diagram.md](rbac_rag_sequence_diagram.md).

## Run it

Prerequisites:

- Python 3.13 or later.
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.
- A local Redis 8 instance with Search and JSON commands available.
- An OpenAI API key for document/query embeddings and live generation.
- The two included PDFs under `resources/`.

From the repository root:

The commands below use `uv` directly. `make setup`, `make doctor`, and `make verify` are optional
aliases. `make redis-start` is an optional Homebrew-oriented launcher for the already-installed
Redis server; you may use your normal service manager instead.

```bash
# Install the locked environment.
uv sync --locked

# Create local configuration, then add OPENAI_API_KEY to .env.
cp .env.example .env

# Optional: start Redis with the repository's Homebrew-oriented wrapper.
make redis-start

# Validate the runtime directly.
uv run portfolio-doctor

# Run ingestion, boundary checks, and grounded chat examples.
uv run python RAG/User_role_based_rag.py
```

The shared `.env` settings used by this example are:

| Variable                                        | Required | Default / behavior                                                  |
| ----------------------------------------------- | -------- | ------------------------------------------------------------------- |
| `OPENAI_API_KEY`                                | Yes      | No default; required by OpenAI vectorization and generation clients |
| `OPENAI_MODEL`                                  | No       | `gpt-5.6-luna`                                                      |
| `OPENAI_EMBEDDING_MODEL`                        | No       | `text-embedding-3-small`                                            |
| `REDIS_URL`                                     | No       | Takes precedence over individual Redis connection fields            |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`          | No       | `localhost`, `6379`, `0`                                            |
| `REDIS_USERNAME`, `REDIS_PASSWORD`, `REDIS_SSL` | No       | Optional authentication and TLS settings                            |
| `REDIS_NAMESPACE`                               | No       | `portfolio`                                                         |

There are no command-line arguments. The users, role mutations, PDFs, queries, threshold, and
chat examples are deliberately fixed to make access-boundary behavior easy to compare.

## Expected output

The run prints user mutations, ingestion counts, expected rejection paths, retrieved result
samples, and the chat-manager responses. IDs, chunk counts, distances, and model wording vary:

```text
Successfully connected to Redis
User 'alice' created.
Retrieved: <User user_id=alice, roles=['finance', 'manager']>
After adding 'executive': ...
After removing 'manager': ...
Extracted ... chunks for doc ... from file ...chevrolet...pdf
Expected error: User tyler not found.
Tyler results (top 3): [...]
Expected error: No available documents found for alice
Extracted ... chunks for doc ... from file ...10-K...pdf
Alice results (top 3): [...]
I couldn't find any relevant documents you have permission to access. ...
...grounded Chevrolet answer...

Sources:
- [1] 2022-chevrolet-commercial-colorado-ebrochure.pdf, page ...
```

On later runs, existing users produce explicit `already exists` messages and are then reloaded.
The PDFs are embedded again, but only the matching content-addressed document keys are replaced.

## Persistence, trust, and limitations

The persistence and trust boundaries below describe how this working demonstration behaves, not
production guarantees. Retained Redis records, role filters, citations, and prompt delimiters are
individual Redis/RAG primitives that require a separately designed production control plane.

1. This demonstrates authorization-filtered retrieval, not authentication. `user_id` is supplied
   directly by the caller, and anyone who can choose another ID or modify Redis may bypass the
   intended application boundary.
2. Filename-based role assignment is illustrative and must not be treated as an authoritative
   classification mechanism.
3. Role matching uses allow tags only. There are no explicit denies, tenant boundaries,
   ownership rules, policy hierarchy, group expansion, or audit log.
4. Within a running manager, per-user history is independent of current roles. Revoking a role
   affects new retrievals but does not delete information already present in that user's prior
   messages. Production role changes should trigger appropriately scoped history review or
   invalidation.
5. The no-document response intentionally does not distinguish between irrelevant content and
   content excluded by permissions, reducing information leakage but also hiding diagnostics
   from the end user.
6. A `0.3` cosine-distance threshold is an example default, not a calibrated guarantee for
   either included corpus.
7. Prompt delimiters and context-only instructions reduce indirect prompt-injection risk; they
   do not replace content scanning, output validation, or provider safety controls.
8. `RAGChatManager.answer()` converts internal exceptions into response text. A production API
   should log sanitized details internally and expose a stable public error instead.
9. Users, documents, vector indexes, and conversation-history records have no configured TTL and
   remain in Redis after the process closes. Because a new RedisVL default `session_tag` is
   generated on restart, old history can remain stored without being selected by the new run.
   Define explicit retention, restoration, and deletion workflows before storing real data.
10. Re-ingestion is scoped but not transactional across delete and load. Concurrent writers for
    the same document ID require coordination.
11. PDFs, prompts, conversation history, and retrieved passages are sent to OpenAI. Apply data
    classification, consent, minimization, encryption, and retention controls before using
    sensitive material.

## Test it

The focused tests use fakes and do not call Redis or OpenAI:

```bash
uv run python -m unittest \
  tests.test_phase2_requirements.RoleBasedRagTests -v
```

The citation-schema integration test requires the configured local Redis instance:

```bash
uv run python -m unittest \
  tests.test_redis_integration.RedisIntegrationTests.test_existing_rag_index_gains_citation_fields_without_data_loss -v
```

Run the complete repository quality gate directly with Redis available:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench
```

`make verify` is the optional convenience alias for these commands.
Full PDF embedding and model generation remain an intentional manual demonstration; the
repository [test strategy](../TESTING.md) makes that boundary explicit.

## License

This project is available under the repository's [MIT License](../LICENSE).
