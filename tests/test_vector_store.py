import os
import pytest
from unittest.mock import patch
from review_assistant.vector_store import VectorStore

@pytest.fixture
def mock_openai_embeddings():
    with patch("review_assistant.vector_store.OpenAIEmbeddings") as mock_embeddings:
        mock_instance = mock_embeddings.return_value
        mock_instance.embed_query.return_value = [0.1] * 1536
        mock_instance.embed_documents.return_value = [[0.1] * 1536]
        yield mock_instance

@pytest.fixture
def vector_store(tmp_path, mock_openai_embeddings):
    index_path = tmp_path / "test_index.faiss"
    return VectorStore(index_path=index_path)

def test_vector_store_initialization(vector_store):
    assert vector_store.index is not None

def test_add_texts_and_save(vector_store):
    vector_store.add_texts(texts=["test text"], metadatas=[{"source": "test"}])
    vector_store.save_index()
    assert os.path.exists(vector_store.index_path)

def test_load_existing_index(tmp_path, mock_openai_embeddings):
    index_path = tmp_path / "test_index.faiss"
    vs1 = VectorStore(index_path=index_path)
    vs1.add_texts(texts=["test text"], metadatas=[{"source": "test"}])
    vs1.save_index()

    vs2 = VectorStore(index_path=index_path)
    assert vs2.index is not None
    retriever = vs2.as_retriever()
    assert retriever is not None