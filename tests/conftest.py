import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True, scope='session')
def mock_vector_store():
    """
    Fixture to mock the VectorStore and its dependencies for the duration of the test session.
    This prevents actual calls to OpenAI's API.
    """
    with patch('review_assistant.api.VectorStore') as mock_vector_store:
        # Ensure that any calls to the retriever return a mock
        mock_vector_store.return_value.as_retriever.return_value = MagicMock()
        yield mock_vector_store