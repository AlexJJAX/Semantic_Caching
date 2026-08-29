# RBAC RAG — End-to-End Request Sequence

This sequence shows how users and documents are prepared, how Redis enforces role filtering
inside vector retrieval, and how the chat manager separates no-evidence, grounded-generation,
and error paths.

```mermaid
sequenceDiagram
    autonumber

    actor Operator
    participant App as Application<br/>main() / run_demo()
    participant Users as RedisJSON<br/>User Records
    participant KB as KnowledgeBase<br/>Ingestion + Search
    participant Embed as OpenAI Embeddings API
    participant Docs as Redis Search + JSON<br/>Document Index
    participant Chat as RAGChatManager
    participant History as RedisVL<br/>Per-User History
    participant LLM as OpenAI Chat Completions<br/>GPT-5.6 Luna

    Note over Operator,LLM: Initialize durable Redis resources

    Operator->>App: Run User_role_based_rag.py
    App->>Users: Create bounded Redis client and PING
    Users-->>App: PONG
    App->>Users: Create or reload alice and larry;<br/>persist normalized roles
    Users-->>App: Current user records
    App->>KB: Construct KnowledgeBase
    KB->>Docs: Create index with overwrite=false
    KB->>Docs: Add source and page fields if missing
    Docs-->>KB: Durable index ready

    Note over App,Docs: Ingest the product and sales corpus

    App->>KB: ingest(Chevrolet brochure)
    KB->>KB: Load PDF, split 512/100,<br/>hash source bytes, assign product and sales
    loop Each brochure chunk
        KB->>Embed: Embed chunk with text-embedding-3-small
        Embed-->>KB: Chunk vector
    end
    KB->>Docs: SCAN and UNLINK only prior keys<br/>for the same document ID
    KB->>Docs: Load JSON chunks with allowed_roles,<br/>content, source, page, and vectors
    Docs-->>KB: Brochure chunks persisted
    KB-->>App: Content-derived document ID

    Note over App,Docs: Direct authorization-boundary checks

    App->>Users: Load tyler before creation
    Users-->>App: No user record
    App-->>Operator: Expected user-not-found error<br/>without retrieval or generation
    App->>Users: Create tyler with sales role
    Users-->>App: User persisted

    App->>Users: Load tyler roles
    Users-->>App: sales
    App->>KB: Search Chevrolet question with sales role
    KB->>Embed: Embed query
    Embed-->>KB: Query vector
    KB->>Docs: VectorRangeQuery with sales TAG filter,<br/>distance at most 0.3, top 5
    Docs-->>KB: Authorized brochure chunks only
    KB-->>App: Source-aware results
    App-->>Operator: Print Tyler result sample

    App->>Users: Load alice roles
    Users-->>App: finance and executive
    App->>KB: Search Chevrolet question with Alice roles
    KB->>Embed: Embed query
    Embed-->>KB: Query vector
    KB->>Docs: VectorRangeQuery with finance or executive filter
    Docs-->>KB: No authorized brochure chunks
    KB-->>App: Empty result
    App-->>Operator: Expected no-document error

    Note over App,Docs: Add the finance and executive corpus

    App->>KB: ingest(Apple 10-K)
    KB->>KB: Load PDF, split 512/100,<br/>hash source bytes, assign finance and executive
    loop Each 10-K chunk
        KB->>Embed: Embed chunk with text-embedding-3-small
        Embed-->>KB: Chunk vector
    end
    KB->>Docs: Replace only prior keys for this document ID
    KB->>Docs: Load source-aware JSON chunks and vectors
    Docs-->>KB: 10-K chunks persisted

    App->>KB: Search Apple revenue with Alice roles
    KB->>Embed: Embed query
    Embed-->>KB: Query vector
    KB->>Docs: Role-filtered vector query
    Docs-->>KB: Authorized 10-K chunks
    KB-->>App: Source-aware results
    App-->>Operator: Print Alice result sample

    Note over App,LLM: RAGChatManager answer path

    App->>Chat: answer(query, user_id)
    Chat->>History: Start or reuse one history index for user_id
    History-->>Chat: Session ready
    Chat->>Users: Reload current user roles
    Users-->>Chat: Validated role set
    Chat->>KB: Search query with current roles
    KB->>Embed: Embed query
    Embed-->>KB: Query vector
    KB->>Docs: Vector query with role and distance filters
    Docs-->>KB: Qualified authorized passages or empty set
    KB-->>Chat: Retrieval result

    alt No qualified authorized passages
        Chat->>History: Store query and permission-safe no-document response
        History-->>Chat: Exchange persisted
        Chat-->>App: Fixed response without an LLM call
    else Authorized passages are available
        Chat->>Chat: Add source/page/chunk labels and<br/>untrusted-passage boundaries
        Chat->>History: Read recent messages for the same user
        History-->>Chat: Session context
        Chat->>LLM: System security policy, history,<br/>question, and authorized evidence
        LLM-->>Chat: Grounded answer with inline citations
        Chat->>Chat: Append source list from the same passages
        Chat->>History: Store query and final answer
        History-->>Chat: Exchange persisted
        Chat-->>App: Grounded answer and sources
    else Lookup, retrieval, or model error
        Chat-->>App: Error response; do not store the failed exchange
    end

    App-->>Operator: Print answer path result

    Note over App,LLM: Close clients; preserve owned application data

    App->>Chat: close()
    Chat->>LLM: Close OpenAI client
    App->>Users: Close Redis connection pool in finally
    Note over Users,History: Users, document index, chunks,<br/>and message histories remain in Redis
    App-->>Operator: Process exits
```
