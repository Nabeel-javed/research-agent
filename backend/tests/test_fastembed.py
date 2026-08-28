import numpy as np
import pytest

from app.config import Settings
from app.embeddings.fastembed import embed_documents, embed_query


class FakeEmbeddingModel:
    def embed(self, texts):
        return (
            np.array([float(index), 1.0], dtype=np.float32)
            for index, _ in enumerate(texts)
        )

    def query_embed(self, query):
        assert query == "current employer"
        return iter([np.array([0.5, 1.0], dtype=np.float32)])


@pytest.mark.asyncio
async def test_uses_document_and_query_specific_fastembed_paths(monkeypatch):
    model = FakeEmbeddingModel()
    monkeypatch.setattr("app.embeddings.fastembed._model", lambda _name: model)
    settings = Settings(fastembed_model="test-model")

    documents = await embed_documents(settings, ["one", "two"])
    query = await embed_query(settings, "current employer")

    assert documents.shape == (2, 2)
    assert documents.dtype == np.float32
    assert query.tolist() == [0.5, 1.0]
