import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings object."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 1536  # Typical embedding size
    mock.embed_documents.return_value = [[0.1] * 1536]
    return mock

@pytest.fixture  
def mock_vector_store():
    """Create a mock vector store."""
    mock = MagicMock()
    mock.as_retriever.return_value = MagicMock()
    return mock
