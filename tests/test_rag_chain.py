import os
import pytest
from unittest.mock import MagicMock, patch
from review_assistant.rag_chain import RAGChain

@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.as_retriever.return_value = MagicMock()
    return mock

@pytest.fixture
def rag_chain(mock_vector_store):
    return RAGChain(vector_store=mock_vector_store, api_key="test_key")

def test_rag_chain_invoke(rag_chain):
    with patch("review_assistant.rag_chain.ChatOpenAI") as mock_chat:
        mock_llm = mock_chat.return_value
        mock_llm.invoke.return_value = "This is a test review."
        
        rag_chain.llm = mock_llm
        rag_chain.chain = rag_chain._build_chain()

        review = rag_chain.invoke("def hello(): pass")
        assert review == "This is a test review."
        rag_chain.vector_store.as_retriever.assert_called()