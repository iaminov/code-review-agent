import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from review_assistant.vector_store import VectorStore

@pytest.fixture
def temp_index_path():
    """Create a temporary path for the index."""
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()

def test_vector_store_initialization(temp_index_path):
    """Test that VectorStore initializes correctly."""
    with patch("review_assistant.vector_store.OpenAIEmbeddings") as mock_embeddings_class:
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        mock_embeddings_class.return_value = mock_embeddings
        
        store = VectorStore(temp_index_path)
        
        assert store.index_path == temp_index_path
        assert store.embeddings == mock_embeddings
        assert store.index is not None

def test_add_texts():
    """Test adding texts to the vector store."""
    with patch("review_assistant.vector_store.OpenAIEmbeddings") as mock_embeddings_class:
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        mock_embeddings_class.return_value = mock_embeddings
        
        store = VectorStore("test.faiss")
        store.index = MagicMock()
        
        texts = ["Hello world", "Test document"]
        metadatas = [{"source": "file1"}, {"source": "file2"}]
        
        store.add_texts(texts, metadatas)
        store.index.add_texts.assert_called_once_with(texts=texts, metadatas=metadatas)

def test_as_retriever():
    """Test getting a retriever from the vector store."""
    with patch("review_assistant.vector_store.OpenAIEmbeddings") as mock_embeddings_class:
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        mock_embeddings_class.return_value = mock_embeddings
        
        store = VectorStore("test.faiss")
        store.index = MagicMock()
        mock_retriever = MagicMock()
        store.index.as_retriever.return_value = mock_retriever
        
        retriever = store.as_retriever()
        assert retriever == mock_retriever
        store.index.as_retriever.assert_called_once()
