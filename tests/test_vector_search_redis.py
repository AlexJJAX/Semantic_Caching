from __future__ import annotations

import importlib.util
import io
import unittest
import uuid
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
from redis.exceptions import RedisError, ResponseError

from redis_ai_portfolio.config import PortfolioSettings
from redis_ai_portfolio.redis import create_redis_client

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_example(relative_path: str) -> ModuleType:
    """Load a runnable example whose directory name is not importable Python syntax."""
    path = REPOSITORY_ROOT / relative_path
    module_name = f"integration_{path.stem.casefold()}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constant_vector(dimensions: int, dtype: str) -> np.ndarray:
    """Return a deterministic unit vector with the requested Redis datatype."""
    vector = np.zeros(dimensions, dtype=dtype)
    vector[0] = 1.0
    return vector


class DeterministicSentenceTransformer:
    """Small embedding stand-in; Redis itself remains real in these tests."""

    def __init__(self, _model: str) -> None:
        pass

    def encode(
        self,
        _text: str,
        *,
        precision: str,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        if precision != "float32" or not convert_to_numpy:
            raise AssertionError("The redis-py example changed its embedding contract")
        return constant_vector(384, "float32")


class DeterministicRedisVLVectorizer:
    """Match RedisVL's vectorizer boundary without downloading model weights."""

    def __init__(self, *, model: str, dtype: str, cache: object) -> None:
        del cache
        self.dimensions = 768 if "mpnet" in model else 384
        self.dtype = dtype

    def embed(self, _content: str, *, as_buffer: bool = False):
        vector = constant_vector(self.dimensions, self.dtype)
        return vector.tobytes() if as_buffer else vector.tolist()

    def embed_many(self, *, contents: list[str], as_buffer: bool = False):
        return [self.embed(content, as_buffer=as_buffer) for content in contents]


class VectorSearchRedisIntegrationTests(unittest.TestCase):
    """Execute each vector-search example against Redis Search end to end."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_settings = PortfolioSettings.from_env()
        cls.client = create_redis_client(cls.base_settings.redis_url)
        try:
            cls.client.ping()
            if not cls.client.execute_command("COMMAND", "INFO", "FT.CREATE"):
                raise unittest.SkipTest("Redis Search is unavailable")
        except RedisError as exc:
            cls.client.close()
            raise unittest.SkipTest(f"Redis integration unavailable: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def setUp(self) -> None:
        self.run_id = uuid.uuid4().hex
        self.namespace = f"vector-integration:{self.run_id}"
        self.settings = replace(self.base_settings, redis_namespace=self.namespace)

    def tearDown(self) -> None:
        keys = list(self.client.scan_iter(match=f"{self.namespace}:*", count=100))
        if keys:
            self.client.unlink(*keys)

    def drop_index(self, index_name: str) -> None:
        try:
            self.client.execute_command("FT.DROPINDEX", index_name, "DD")
        except ResponseError:
            pass

    def test_redispy_example_executes_real_search_and_aggregation_queries(self) -> None:
        example = load_example("vector_search/1_redispy/Redispy.py")
        example.SETTINGS = self.settings
        example.INDEX_NAME = self.settings.redis_name("idx", "redispy")
        example.KEY_PREFIX = f"{self.settings.redis_name('movie', 'redispy')}:"
        output = io.StringIO()
        try:
            with (
                patch.object(
                    example,
                    "SentenceTransformer",
                    DeterministicSentenceTransformer,
                ),
                redirect_stdout(output),
            ):
                example.run_demo(self.client)
        finally:
            self.drop_index(example.INDEX_NAME)

        rendered = output.getvalue()
        self.assertIn("Index loaded with", rendered)
        self.assertIn("Basic KNN", rendered)
        self.assertIn("Aggregation: avg rating per genre", rendered)
        self.assertIn("Demo index and keys removed", rendered)

    def test_redisvl_example_executes_real_vector_text_and_hybrid_queries(self) -> None:
        example = load_example("vector_search/2_redisvl/Redisvl.py")
        example.SETTINGS = self.settings
        example.INDEX_NAME = self.settings.redis_name("idx", "redisvl")
        example.KEY_PREFIX = self.settings.redis_name("movie", "redisvl")
        output = io.StringIO()
        try:
            with (
                patch.object(
                    example,
                    "HFTextVectorizer",
                    DeterministicRedisVLVectorizer,
                ),
                redirect_stdout(output),
            ):
                example.run_demo(self.client)
        finally:
            self.drop_index(example.INDEX_NAME)

        rendered = output.getvalue()
        self.assertIn("Standard KNN vector search", rendered)
        self.assertIn("Full-text BM25 search", rendered)
        self.assertIn("Hybrid search", rendered)
        self.assertIn("Index deleted", rendered)

    def test_multivector_example_executes_real_weighted_redis_query(self) -> None:
        example = load_example(
            "vector_search/3_multivector_search/Multivector_search.py"
        )
        example.SETTINGS = self.settings
        example.INDEX_NAME = self.settings.redis_name("idx", "multivector")
        example.KEY_PREFIX = self.settings.redis_name("movie", "multivector")
        output = io.StringIO()
        try:
            with (
                patch.object(
                    example,
                    "HFTextVectorizer",
                    DeterministicRedisVLVectorizer,
                ),
                redirect_stdout(output),
            ):
                example.run_demo(self.client)
        finally:
            self.drop_index(example.INDEX_NAME)

        rendered = output.getvalue()
        self.assertIn("Generated embeddings", rendered)
        self.assertIn("Multi-vector search", rendered)
        self.assertIn("combined_score", rendered)
        self.assertIn("Index deleted", rendered)


if __name__ == "__main__":
    unittest.main()
