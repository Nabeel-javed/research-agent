"""FastEmbed-backed document and query embeddings.

The ONNX model runs inside the backend process, so document retrieval does not
depend on LM Studio. Model files are downloaded and cached by FastEmbed on the
first real embedding request.
"""

from __future__ import annotations

import asyncio
from threading import Lock

import numpy as np
from fastembed import TextEmbedding

from app.config import Settings


class EmbeddingError(RuntimeError):
    """The local embedding model could not produce usable vectors."""


_models: dict[str, TextEmbedding] = {}
_model_load_lock = Lock()


def _model(model_name: str) -> TextEmbedding:
    # Several upload workers may reach the first request together. Load and
    # cache exactly one model instance before allowing concurrent inference.
    with _model_load_lock:
        model = _models.get(model_name)
        if model is None:
            model = TextEmbedding(model_name=model_name)
            _models[model_name] = model
        return model


def _documents_sync(model_name: str, texts: list[str]) -> np.ndarray:
    vectors = list(_model(model_name).embed(texts))
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or len(array) != len(texts):
        raise EmbeddingError("FastEmbed returned an invalid document embedding batch.")
    return array


def _query_sync(model_name: str, query: str) -> np.ndarray:
    vectors = list(_model(model_name).query_embed(query))
    if len(vectors) != 1:
        raise EmbeddingError("FastEmbed returned an invalid query embedding.")
    vector = np.asarray(vectors[0], dtype=np.float32)
    if vector.ndim != 1:
        raise EmbeddingError("FastEmbed returned an invalid query embedding.")
    return vector


async def embed_documents(settings: Settings, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    try:
        return await asyncio.to_thread(_documents_sync, settings.fastembed_model, texts)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("FastEmbed document embedding failed.") from exc


async def embed_query(settings: Settings, query: str) -> np.ndarray:
    try:
        return await asyncio.to_thread(_query_sync, settings.fastembed_model, query)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("FastEmbed query embedding failed.") from exc
