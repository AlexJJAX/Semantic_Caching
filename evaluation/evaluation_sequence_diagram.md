# Redis RAG Evaluation — End-to-End Sequence

This sequence shows the optional synthetic-testset workflow followed by the evaluation run.
The evaluation path retrieves once per question and reuses the same documents for generation
and Ragas scoring.

```mermaid
sequenceDiagram
    autonumber

    actor Engineer
    participant BatchCLI as Testset CLI<br/>generate_testset.py
    participant PDF as Source PDF
    participant Batch as OpenAI Batch API
    participant State as Atomic Run Manifest<br/>.batch_state.json
    participant Synth as Ragas Testset Generator
    participant Eval as Evaluation Runner<br/>06_ragas_evaluation.py
    participant Embed as OpenAI Embeddings API
    participant Redis as Redis Search<br/>Temporary Vector Index
    participant LLM as GPT-5.6 Luna
    participant Judge as Ragas Evaluators
    participant Artifacts as CSV Artifacts

    Note over Engineer,Artifacts: Optional synthetic testset preparation

    Engineer->>BatchCLI: --submit --pdf source.pdf --state manifest.json
    BatchCLI->>PDF: Load pages and split into exact chunks
    PDF-->>BatchCLI: Chunks with source and page metadata
    BatchCLI->>BatchCLI: Build temporary embeddings JSONL
    BatchCLI->>Batch: Upload JSONL and create /v1/embeddings batch
    Batch-->>BatchCLI: File ID, batch ID, and status
    BatchCLI->>State: Atomically persist source hash, exact chunks,<br/>models, generation settings, and batch identity
    BatchCLI->>BatchCLI: Delete temporary JSONL and close API client
    BatchCLI-->>Engineer: Batch ID and manifest location

    Engineer->>BatchCLI: --collect [--partial | --no-poll]
    BatchCLI->>State: Load exact submitted chunks and run configuration

    loop While polling and the batch is non-terminal
        BatchCLI->>Batch: Retrieve batch status
        Batch-->>BatchCLI: Progress counters and current status
    end

    alt Complete output is available
        BatchCLI->>Batch: Download output file
        Batch-->>BatchCLI: Embeddings keyed by chunk custom_id
    else Partial output is available
        BatchCLI->>Batch: Download exposed partial output
        Batch-->>BatchCLI: Available subset of chunk embeddings
    else No output is available in partial mode
        BatchCLI->>BatchCLI: Start with an empty embedding cache
    end

    BatchCLI->>State: Append collection attempt and Batch coverage
    BatchCLI->>Synth: Generate single-hop questions from saved chunks

    loop For each embedding absent from Batch output
        Synth->>Embed: Request live document or query embedding
        Embed-->>Synth: Embedding vector
    end

    Synth->>LLM: Generate synthetic questions and references
    LLM-->>Synth: Test samples
    Synth-->>BatchCLI: Synthetic testset
    BatchCLI->>Artifacts: Atomically write new_testset.csv
    BatchCLI->>State: Record completion, row count, and output path
    BatchCLI->>BatchCLI: Close embedding client in finally
    BatchCLI-->>Engineer: Testset path and Batch coverage

    Note over Engineer,Artifacts: Redis-backed retrieve-once evaluation

    Engineer->>Eval: Run 06_ragas_evaluation.py
    Eval->>Redis: Drop only the namespaced evaluation index if present
    Eval->>PDF: Load pages and split into 512/100 chunks
    PDF-->>Eval: Documents with source and page metadata

    loop For each source chunk
        Eval->>Embed: Create text-embedding-3-small vector
        Embed-->>Eval: Chunk vector
        Eval->>Redis: Store document and vector in temporary index
    end

    Eval->>Artifacts: Read new_testset.csv

    loop For each testset row
        Eval->>Embed: Embed the question once for retrieval
        Embed-->>Eval: Query vector
        Eval->>Redis: Vector search, k=4, score threshold 0.7
        Redis-->>Eval: Qualified Documents D with source/page metadata
        Note over Eval: Preserve raw Documents D
        Eval->>LLM: Question plus D as delimited untrusted evidence
        LLM-->>Eval: Grounded answer with numbered citations
        Eval->>Eval: Store question, answer, reference,<br/>and contexts extracted from the same D
    end

    Eval->>Judge: Score faithfulness and answer relevancy
    Judge->>LLM: LLM-based generation judgments
    LLM-->>Judge: Per-row scores
    Judge->>Embed: Embeddings required by answer relevancy
    Embed-->>Judge: Evaluation vectors

    Eval->>Judge: Score context recall and context precision
    Judge->>LLM: LLM-based retrieval judgments
    LLM-->>Judge: Per-row scores
    Judge-->>Eval: Four metric columns

    Eval->>Artifacts: Write metrics_512_100.csv
    Eval->>Redis: Disconnect vector store and drop<br/>the namespaced index and documents in finally
    Eval-->>Engineer: Metric summaries and output path
```
