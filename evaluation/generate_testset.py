"""
Synthetic Testset Generator (ragas) — with OpenAI Batch API for embeddings

Splits testset generation into two decoupled phases so that the expensive
document-embedding step runs asynchronously via the Batch API (50 % cost
reduction, higher rate limits, up-to-24 h SLA).

Phase 1  --submit
    • Load PDF → chunk it
    • Write a JSONL batch file (one /v1/embeddings request per chunk)
    • Upload the file to OpenAI and create a Batch job
    • Save the batch_id + chunk texts to a state file, then exit

Phase 2  --collect [--batch-id BATCH_ID]
    • Retrieve (or poll for) the completed batch results
    • Re-hydrate a CachedEmbeddings object from the batch output
    • Run ragas TestsetGenerator with those pre-computed embeddings
    • Save the testset to CSV

Usage:
    # Submit the embedding job (returns immediately)
    python evaluation/generate_testset.py --submit \\
        --pdf  resources/nke-10k-2023.pdf \\
        --state evaluation/.batch_state.json

    # Later: collect results and generate testset
    python evaluation/generate_testset.py --collect \\
        --state evaluation/.batch_state.json \\
        --out   evaluation/new_testset.csv \\
        --size  10

Dependencies: openai>=1.0, langchain-openai, langchain-community,
              langchain-text-splitters, langchain-core, ragas, pypdf,
              python-dotenv
"""

# --- Stdlib ---
import argparse
import hashlib
import json
import os
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# --- Third-party ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

# Use only the single-hop synthesizer. This keeps the generated dataset focused and
# avoids schema-parsing failures previously observed with smaller models.
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

from redis_ai_portfolio.config import get_settings

SETTINGS = get_settings()

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore")


# --- Constants / Defaults ---

DEFAULT_PDF_PATH   = "resources/nke-10k-2023.pdf"
DEFAULT_STATE_PATH = "evaluation/.batch_state.json"
DEFAULT_OUT_PATH   = "evaluation/new_testset.csv"
DEFAULT_TEST_SIZE  = 10
DEFAULT_CHUNK_SIZE    = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHARS = 400
STATE_SCHEMA_VERSION = 2
BATCH_ENDPOINT = "/v1/embeddings"
BATCH_COMPLETION_WINDOW = "24h"
GENERATION_RUN_CONFIG = {
    "timeout": 200,
    "max_wait": 160,
    "max_retries": 3,
}

EMBEDDING_MODEL = SETTINGS.openai_embedding_model
# ragas 0.4 unified generator + critic into a single LLM (no separate CRITIC_MODEL)
GENERATOR_MODEL = SETTINGS.openai_model


# How often to poll the batch status (seconds)
POLL_INTERVAL = 30
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
STALL_WARN_AFTER_POLLS = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: str, payload: dict) -> None:
    """Replace a JSON state file atomically and clean its temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tmp",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
            encoding="utf-8",
        ) as temporary:
            json.dump(payload, temporary, indent=2)
            temporary.write("\n")
            temp_path = temporary.name
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _write_csv_atomic(path: str, dataframe) -> None:
    """Replace an output CSV atomically and clean its temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
            encoding="utf-8",
        ) as temporary:
            temp_path = temporary.name
            dataframe.to_csv(temporary, index=False)
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _batch_output_file(batch, *, partial: bool) -> str | None:
    """Choose completed/partial output; None means use the live fallback cache."""
    if batch.status == "completed":
        if not batch.output_file_id:
            raise RuntimeError(f"Completed batch {batch.id} has no output file")
        return batch.output_file_id
    if partial:
        return batch.output_file_id
    raise RuntimeError(f"Batch {batch.id} is not complete (status: {batch.status})")


def wait_for_batch(
    client,
    batch_id: str,
    *,
    poll: bool,
    partial: bool,
    expected_requests: int,
    poll_interval: int = POLL_INTERVAL,
):
    """Return a complete batch or stop immediately when partial collection is requested."""
    last_completed = -1
    stall_count = 0

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        completed = counts.completed if counts else 0
        total = counts.total if counts else expected_requests
        failed = counts.failed if counts else 0
        print(
            f"[collect] Status: {batch.status}  "
            f"({completed}/{total} completed, {failed} failed)"
        )

        if batch.status == "completed":
            return batch
        if partial:
            print(
                "[collect] --partial requested; collecting available output and "
                "using live embedding fallback for any missing chunks."
            )
            return batch
        if batch.status in TERMINAL_BATCH_STATUSES:
            raise RuntimeError(
                f"Batch {batch_id} ended with status '{batch.status}'. "
                f"Check errors at: https://platform.openai.com/batches/{batch_id}"
            )
        if not poll:
            raise RuntimeError(
                f"Batch {batch_id} not yet complete (status: '{batch.status}'). "
                "Re-run with --collect to resume polling."
            )

        if completed == last_completed:
            stall_count += 1
        else:
            stall_count = 0
            last_completed = completed

        if stall_count >= STALL_WARN_AFTER_POLLS:
            stuck = total - completed - failed
            print(
                f"\n[collect] Progress stalled for "
                f"{stall_count * poll_interval}s; {stuck} request(s) have not moved.\n"
                f"           Monitor: https://platform.openai.com/batches/{batch_id}\n"
                "           Re-run with --partial to stop polling and use live "
                "fallback for unavailable results.\n"
            )

        print(f"[collect] Waiting {poll_interval}s before next check...")
        time.sleep(poll_interval)


def build_submission_state(
    *,
    batch,
    uploaded_file_id: str,
    pdf_path: str,
    chunks: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """Capture the complete, reproducible configuration for a submitted run."""
    source_path = Path(pdf_path)
    now = _utc_now()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "status": "submitted",
        "source": {
            "pdf_path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "chunking": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks),
        },
        "models": {
            "embedding": EMBEDDING_MODEL,
            "generator": GENERATOR_MODEL,
        },
        "batch": {
            "id": batch.id,
            "input_file_id": uploaded_file_id,
            "endpoint": BATCH_ENDPOINT,
            "completion_window": BATCH_COMPLETION_WINDOW,
            "status": batch.status,
        },
        "generation": {
            "synthesizer": "single-hop-specific",
            "query_distribution": 1.0,
            "minimum_chunk_characters": DEFAULT_MIN_CHARS,
            "run_config": dict(GENERATION_RUN_CONFIG),
            "raise_exceptions": False,
        },
        "chunks": [
            {"content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ],
        "collections": [],
    }


# ---------------------------------------------------------------------------
# CachedEmbeddings — LangChain Embeddings adapter backed by a pre-built dict
# ---------------------------------------------------------------------------

class CachedEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation that serves vectors from a
    pre-computed lookup dictionary instead of calling the OpenAI API.

    The cache maps chunk text → embedding vector (List[float]).  Any text
    not found in the cache falls back to the live OpenAI API, so ragas
    internal queries (e.g. question-to-chunk similarity) still work.
    """

    def __init__(self, cache: Dict[str, List[float]], model: str = EMBEDDING_MODEL):
        self._cache = cache
        self._model = model
        self._client = OpenAI(api_key=SETTINGS.openai_api_key)

    def close(self) -> None:
        """Release the fallback embedding HTTP client."""
        self._client.close()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return cached embeddings; fall back to live API for cache misses."""
        results = []
        missed = [(i, t) for i, t in enumerate(texts) if t not in self._cache]

        if missed:
            # Batch-fetch only the missing texts via the live API
            indices, miss_texts = zip(*missed)
            response = self._client.embeddings.create(
                model=self._model,
                input=list(miss_texts),
            )
            for (idx, text), item in zip(missed, response.data):
                self._cache[text] = item.embedding

        for text in texts:
            results.append(self._cache[text])
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (always uses live API — low volume)."""
        if text not in self._cache:
            response = self._client.embeddings.create(
                model=self._model,
                input=[text],
            )
            self._cache[text] = response.data[0].embedding
        return self._cache[text]


# ---------------------------------------------------------------------------
# Phase 1 — Submit
# ---------------------------------------------------------------------------

def submit_batch(
    pdf_path: str,
    state_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> str:
    """
    Chunk the PDF, build an embedding JSONL, upload it, and create a Batch job.

    The batch_id and chunk texts are persisted to *state_path* so that
    :func:`collect_batch` can resume without re-reading the PDF.

    Args:
        pdf_path:     Path to the source PDF.
        state_path:   JSON file where the batch_id and chunks are saved.
        chunk_size:   Character size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        The OpenAI batch_id string.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    # --- Load and chunk ---
    print(f"[submit] Loading PDF: {pdf_path}")
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    chunk_texts = [c.page_content for c in chunks]
    print(f"[submit] Created {len(chunk_texts)} chunks")

    # --- Build JSONL batch file ---
    # Each line: one /v1/embeddings request for a single chunk text.
    # We use the chunk index as custom_id so we can reassemble order later.
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            for i, text in enumerate(chunk_texts):
                record = {
                    "custom_id": f"chunk-{i}",
                    "method": "POST",
                    "url": BATCH_ENDPOINT,
                    "body": {
                        "model": EMBEDDING_MODEL,
                        "input": text,
                        "encoding_format": "float",
                    },
                }
                tmp.write(json.dumps(record) + "\n")
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"[submit] JSONL batch file written ({len(chunk_texts)} requests)")

    # --- Upload file and create the Batch job ---
    client = None
    uploaded = None
    batch = None
    try:
        client = OpenAI(api_key=SETTINGS.openai_api_key)
        with open(tmp_path, "rb") as batch_file:
            uploaded = client.files.create(file=batch_file, purpose="batch")
        print(f"[submit] Uploaded file: {uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint=BATCH_ENDPOINT,
            completion_window=BATCH_COMPLETION_WINDOW,
            metadata={"description": f"ragas testset embeddings — {pdf_path}"},
        )
    except Exception:
        if client is not None and uploaded is not None and batch is None:
            try:
                client.files.delete(uploaded.id)
            except Exception:
                pass
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if client is not None:
            client.close()

    print(f"[submit] Batch created: {batch.id}  (status: {batch.status})")
    print("[submit] Results will be ready within 24 h (often much faster).")

    # --- Persist state ---
    state = build_submission_state(
        batch=batch,
        uploaded_file_id=uploaded.id,
        pdf_path=pdf_path,
        chunks=chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    _write_json_atomic(state_path, state)
    print(f"[submit] State saved to: {state_path}")

    return batch.id


# ---------------------------------------------------------------------------
# Phase 2 — Collect
# ---------------------------------------------------------------------------

def collect_batch(
    state_path: str,
    output_path: str,
    test_size: int = DEFAULT_TEST_SIZE,
    batch_id: str = None,
    poll: bool = True,
    partial: bool = False,
) -> None:
    """
    Poll for a completed Batch, download results, and run ragas testset generation.

    Args:
        state_path:  JSON file written by :func:`submit_batch`.
        output_path: Destination CSV for the generated testset.
        test_size:   Number of synthetic test samples to generate.
        batch_id:    Override the batch_id from the state file (optional).
        poll:        If True, block and poll until the batch is complete.
                     If False, raise immediately if the batch is not yet done.
        partial:     If True and the batch is stalled/in-progress, download
                     whatever output_file_id is already available and proceed
                     with the completed subset. Missing chunks fall back to the
                     live API inside CachedEmbeddings.
    """
    # --- Load state ---
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"State file not found: {state_path}\n"
            "Run with --submit first to create a batch job."
        )
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    batch_state = state.get("batch", {})
    source_state = state.get("source", {})
    models_state = state.get("models", {})
    chunk_records = state.get("chunks")

    bid = batch_id or batch_state.get("id") or state.get("batch_id")
    if not bid:
        raise ValueError("Batch state does not contain a batch ID")
    pdf_path = source_state.get("pdf_path") or state.get("pdf_path")
    if not pdf_path:
        raise ValueError("Batch state does not contain a PDF path")
    if chunk_records:
        chunk_texts = [record["content"] for record in chunk_records]
    else:
        chunk_texts = state.get("chunk_texts", [])
        chunk_records = [
            {
                "content": text,
                "metadata": {"source": pdf_path, "chunk_index": index},
            }
            for index, text in enumerate(chunk_texts)
        ]
    if not chunk_texts:
        raise ValueError("Batch state does not contain document chunks")

    embedding_model = models_state.get("embedding") or state.get(
        "embedding_model", EMBEDDING_MODEL
    )
    generator_model = models_state.get("generator", GENERATOR_MODEL)

    print(f"[collect] Checking batch: {bid}")
    client = OpenAI(api_key=SETTINGS.openai_api_key)

    try:
        batch = wait_for_batch(
            client,
            bid,
            poll=poll,
            partial=partial,
            expected_requests=len(chunk_texts),
        )
        output_file_id = _batch_output_file(batch, partial=partial)
        if output_file_id:
            print(f"[collect] Downloading results from file: {output_file_id}")
            raw = client.files.content(output_file_id).text
        else:
            print("[collect] No partial output file is available; using live fallback.")
            raw = ""
    finally:
        client.close()

    # Build text → embedding cache from batch output
    # Output order is NOT guaranteed — use custom_id to reassemble
    index_to_embedding: Dict[int, List[float]] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            print(f"[collect] WARNING: failed request {row['custom_id']}: {row['error']}")
            continue
        custom_id = row.get("custom_id", "")
        if not custom_id.startswith("chunk-"):
            raise ValueError(f"Unexpected custom_id on output line {line_number}: {custom_id!r}")
        idx = int(custom_id.removeprefix("chunk-"))
        if not 0 <= idx < len(chunk_texts):
            raise ValueError(f"Chunk index {idx} is outside saved state")
        response = row.get("response") or {}
        if response.get("status_code") not in {None, 200}:
            print(
                f"[collect] WARNING: request {custom_id} returned "
                f"HTTP {response.get('status_code')}"
            )
            continue
        embedding = response["body"]["data"][0]["embedding"]
        index_to_embedding[idx] = embedding

    # Map chunk text → embedding vector
    cache: Dict[str, List[float]] = {}
    for i, text in enumerate(chunk_texts):
        if i in index_to_embedding:
            cache[text] = index_to_embedding[i]
        else:
            print(f"[collect] WARNING: no embedding for chunk {i} — will re-embed live")

    print(
        f"[collect] Loaded {len(cache)}/{len(chunk_texts)} embeddings from batch results"
    )

    # Reuse the exact chunks and metadata persisted at submission time.
    chunks = [
        Document(
            page_content=record["content"],
            metadata=record.get("metadata") or {"source": pdf_path},
        )
        for record in chunk_records
    ]

    # ragas requires documents to be longer than ~100 tokens.
    # Filter out very short chunks (page headers, footers, single-line snippets)
    # that would cause: "Documents appears to be too short (ie 100 tokens or less)."
    minimum_chars = state.get("generation", {}).get(
        "minimum_chunk_characters",
        DEFAULT_MIN_CHARS,
    )
    full_chunks = chunks
    chunks = [c for c in full_chunks if len(c.page_content.strip()) >= minimum_chars]
    dropped = len(full_chunks) - len(chunks)
    if dropped:
        print(f"[collect] Filtered out {dropped} short chunks (<{minimum_chars} chars); "
              f"{len(chunks)} remain for testset generation")

    run_config_values = dict(
        state.get("generation", {}).get("run_config", GENERATION_RUN_CONFIG)
    )
    collection_run = {
        "started_at": _utc_now(),
        "status": "generating",
        "batch_id": bid,
        "batch_status": batch.status,
        "output_file_id": output_file_id,
        "partial": partial,
        "poll": poll,
        "requested_test_size": test_size,
        "output_path": output_path,
        "embedding_model": embedding_model,
        "generator_model": generator_model,
        "available_batch_embeddings": len(cache),
        "total_chunks": len(chunk_texts),
        "generation_chunks": len(chunks),
        "minimum_chunk_characters": minimum_chars,
        "run_config": run_config_values,
        "synthesizer": "single-hop-specific",
        "query_distribution": 1.0,
        "raise_exceptions": False,
    }
    state["schema_version"] = STATE_SCHEMA_VERSION
    state["updated_at"] = _utc_now()
    state["status"] = "collecting"
    state.setdefault("batch", {}).update(
        {
            "id": bid,
            "status": batch.status,
            "output_file_id": output_file_id,
        }
    )
    state.setdefault("collections", []).append(collection_run)
    _write_json_atomic(state_path, state)

    cached_embeddings = CachedEmbeddings(cache=cache, model=embedding_model)
    try:
        generator_llm = ChatOpenAI(
            model=generator_model,
            api_key=SETTINGS.openai_api_key,
        )
        run_config = RunConfig(**run_config_values)
        generator = TestsetGenerator.from_langchain(
            generator_llm,
            cached_embeddings,
        )
        single_hop = SingleHopSpecificQuerySynthesizer(llm=generator.llm)
        query_distribution = [(single_hop, 1.0)]

        print(f"[collect] Generating {test_size} synthetic test samples (single-hop)...")
        testset = generator.generate_with_langchain_docs(
            chunks,
            testset_size=test_size,
            query_distribution=query_distribution,
            run_config=run_config,
            raise_exceptions=False,
        )
        dataframe = testset.to_pandas()
        _write_csv_atomic(output_path, dataframe)

        collection_run.update(
            {
                "status": "completed",
                "completed_at": _utc_now(),
                "generated_rows": len(dataframe),
            }
        )
        state["status"] = "collected"
        print(f"[collect] Testset saved to: {output_path}  ({len(dataframe)} rows)")
    except Exception as exc:
        collection_run.update(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        state["status"] = "collection_failed"
        raise
    finally:
        cached_embeddings.close()
        state["updated_at"] = _utc_now()
        _write_json_atomic(state_path, state)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic ragas testset using OpenAI Batch API for embeddings.\n\n"
            "Two-phase workflow:\n"
            "  1. --submit   Chunk the PDF, submit embedding batch, save state file.\n"
            "  2. --collect  Poll batch, download embeddings, run ragas, save CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mutually exclusive phase flags
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--submit",  action="store_true", help="Submit the embedding batch job and exit")
    phase.add_argument("--collect", action="store_true", help="Poll batch results and generate the testset")

    # Shared options
    parser.add_argument("--state",  default=DEFAULT_STATE_PATH,
                        help=f"Path to the batch state JSON file (default: {DEFAULT_STATE_PATH})")

    # Submit-only options
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH,
                        help=f"Source PDF path (--submit only, default: {DEFAULT_PDF_PATH})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Chunk size for text splitting (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"Chunk overlap for text splitting (default: {DEFAULT_CHUNK_OVERLAP})")

    # Collect-only options
    parser.add_argument("--out",  default=DEFAULT_OUT_PATH,
                        help=f"Output CSV path for the testset (--collect only, default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--size", type=int, default=DEFAULT_TEST_SIZE,
                        help=f"Number of test samples to generate (default: {DEFAULT_TEST_SIZE})")
    parser.add_argument("--batch-id", default=None,
                        help="Override batch_id from the state file (--collect only)")
    parser.add_argument("--no-poll", action="store_true",
                        help="Fail immediately if batch is not yet complete (--collect only)")
    parser.add_argument("--partial", action="store_true",
                        help=(
                            "Stop polling and use any available batch output immediately "
                            "(--collect only); missing chunks fall back to the live API."
                        ))

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.submit:
        submit_batch(
            pdf_path=args.pdf,
            state_path=args.state,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:  # --collect
        collect_batch(
            state_path=args.state,
            output_path=args.out,
            test_size=args.size,
            batch_id=args.batch_id,
            poll=not args.no_poll,
            partial=args.partial,
        )
