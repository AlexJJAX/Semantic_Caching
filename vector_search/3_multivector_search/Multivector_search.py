"""
Multi-Vector Search with RedisVL

Demonstrates how to build a multi-vector search index in Redis using multiple
HuggingFace embedding models, each capturing a different semantic perspective
of the same movie data. Covers:
  1. Loading movie data from JSON
  2. Generating multiple embeddings per document with different HFTextVectorizers
     (backed by EmbeddingsCache) and different dimensionalities / dtypes
  3. Defining a multi-vector IndexSchema with three vector fields
  4. Using MultiVectorQuery with per-field weights to blend similarity scores

Key concept: each vector field may have a different model, dimensionality, and
dtype — but all must use cosine distance_metric for proper relative weighting.

Dependencies: redisvl>=0.16.0, sentence-transformers, python-dotenv
Resources: place resources/movies.json in the working directory.
Redis must be running (default: localhost:6379).
"""

# --- Stdlib ---
import json
import os
import warnings

# --- Third-party ---
from redis import Redis
from redis.exceptions import ResponseError
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.index import SearchIndex
from redisvl.query import MultiVectorQuery, Vector
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import HFTextVectorizer

from redis_ai_portfolio.config import get_settings
from redis_ai_portfolio.redis import create_redis_client

# Suppress noisy third-party warnings (tokenizer parallelism etc.)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Redis Connection ---

SETTINGS = get_settings()
REDIS_URL = SETTINGS.redis_url


# --- Constants ---

INDEX_NAME = SETTINGS.redis_name("idx", "movies", "multivector")
KEY_PREFIX = SETTINGS.redis_name("movie", "multivector")


# --- Main Demo ---

def run_demo(client: Redis) -> None:
    """Run the multi-vector queries with an initialized Redis client."""

    # --- Load movie dataset ---
    with open("resources/movies.json", "r") as f:
        movies = json.load(f)
    print(f"Loaded {len(movies)} movies.")

    # --- Embed movie descriptions ---
    # Three embedding models, each capturing a different semantic perspective.
    # All use EmbeddingsCache to avoid re-computing embeddings across runs.
    # Note: different models may have different output dimensions and dtypes —
    # this is intentional and fully supported by MultiVectorQuery.
    #
    # redisvl 0.16: Vector also supports max_distance (0.0–2.0) to apply a
    # per-field distance threshold cutoff — useful when you want to exclude
    # semantically distant results from a specific field.

    # Model 1: General-purpose embeddings (384-dim, float64)
    general_model = HFTextVectorizer(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dtype="float64",
        cache=EmbeddingsCache(
            name=SETTINGS.redis_name("cache", "embeddings", "multivector", "general"),
            ttl=600,
            redis_client=client,
        ),
    )

    # Model 2: Movie-specific embeddings — captures richer description semantics (768-dim, float32)
    movie_model = HFTextVectorizer(
        model="sentence-transformers/all-mpnet-base-v2",
        dtype="float32",
        cache=EmbeddingsCache(
            name=SETTINGS.redis_name("cache", "embeddings", "multivector", "movie"),
            ttl=600,
            redis_client=client,
        ),
    )

    # Model 3: Genre-aware embeddings — description prefixed with genre tag (384-dim, float32)
    genre_model = HFTextVectorizer(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dtype="float32",
        cache=EmbeddingsCache(
            name=SETTINGS.redis_name("cache", "embeddings", "multivector", "genre"),
            ttl=600,
            redis_client=client,
        ),
    )

    # Generate multiple embeddings per movie.
    # Each document gets three separate vector fields from three different models.
    # The data source for each field can be anything (description, genre+description, title, etc.)
    print("Generating multi-vector embeddings for movies...")
    multi_vector_data = []
    for movie in movies:
        multi_vector_data.append({
            **movie,
            # General description embedding
            "description_vector_general": general_model.embed(
                movie["description"], as_buffer=True
            ),
            # Movie-specific description embedding
            "description_vector_movie": movie_model.embed(
                movie["description"], as_buffer=True
            ),
            # Genre-prefixed description embedding (improves genre-aware retrieval)
            "description_vector_genre": genre_model.embed(
                f"{movie['genre']} {movie['description']}", as_buffer=True
            ),
        })
    print(f"Generated embeddings for {len(multi_vector_data)} movies.\n")

    # --- Define multi-vector index schema ---
    # Each vector field has its own dims, algorithm, and datatype.
    # All vector fields MUST use cosine distance_metric for proper score normalisation.
    schema = IndexSchema.from_dict({
        "index": {
            "name": INDEX_NAME,
            "prefix": KEY_PREFIX,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title",       "type": "text"},
            {"name": "description", "type": "text"},
            {"name": "genre",   "type": "tag",     "attrs": {"sortable": True}},
            {"name": "rating",  "type": "numeric", "attrs": {"sortable": True}},
            {
                "name": "description_vector_general",
                "type": "vector",
                "attrs": {
                    "dims": 384,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float64",
                },
            },
            {
                "name": "description_vector_movie",
                "type": "vector",
                "attrs": {
                    "dims": 768,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32",
                },
            },
            {
                "name": "description_vector_genre",
                "type": "vector",
                "attrs": {
                    "dims": 384,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32",
                },
            },
        ],
    })

    # --- Create index and load data ---
    # validate_on_load=True checks that loaded records match the schema at write time
    index = SearchIndex(schema, redis_client=client, validate_on_load=True)
    index.create(overwrite=True, drop=True)
    index.load(multi_vector_data)
    print(f"Index '{INDEX_NAME}' created and loaded with {len(multi_vector_data)} records.\n")

    # --- Multi-vector query ---
    # MultiVectorQuery combines similarity scores from multiple vector fields.
    # Each Vector object specifies:
    #   - the query embedding (as bytes buffer)
    #   - the field to search against
    #   - the dtype (must match the index field datatype)
    #   - the weight (relative importance; weights are normalised internally)
    query_text = "action movie with superheroes and explosions"
    num_results = 5

    query_vectors = [
        Vector(
            vector=general_model.embed(query_text, as_buffer=True),
            field_name="description_vector_general",
            dtype="float64",
            weight=0.3,   # 30% — general semantic similarity
        ),
        Vector(
            vector=movie_model.embed(query_text, as_buffer=True),
            field_name="description_vector_movie",
            dtype="float32",
            weight=0.5,   # 50% — richer movie-specific semantics
        ),
        Vector(
            vector=genre_model.embed(query_text, as_buffer=True),  # no f-string needed
            field_name="description_vector_genre",
            dtype="float32",
            weight=0.2,   # 20% — genre-aware perspective
        ),
    ]

    query = MultiVectorQuery(
        vectors=query_vectors,
        num_results=num_results,
        return_fields=["title", "description", "genre", "rating"],
    )

    # MultiVectorQuery result dict includes:
    #   combined_score  — the final weighted blend (higher = more relevant)
    #   score_0/1/2     — per-field normalised similarity scores (0–1)
    #   distance_0/1/2  — raw per-field cosine distances
    #   + all return_fields requested
    print(f"--- Multi-vector search: '{query_text}' (top {num_results}) ---")
    results = index.query(query)
    for i, result in enumerate(results, 1):
        combined = float(result.get("combined_score", 0))
        print(f"{i}. {result['title']}  [combined_score: {combined:.4f}]")
        print(f"   Genre: {result['genre']}, Rating: {result['rating']}")
        print(f"   Description: {result['description'][:100]}...")
        print()

    # --- Cleanup ---
    index.delete()
    print("Index deleted. Done.")


def main() -> None:
    """Connect, run the demo, and clean up its namespace even after failures."""
    client = create_redis_client(REDIS_URL)
    try:
        client.ping()
        print("Connected to Redis.\n")
        run_demo(client)
    except Exception:
        try:
            client.execute_command("FT.DROPINDEX", INDEX_NAME, "DD")
        except ResponseError:
            pass
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
