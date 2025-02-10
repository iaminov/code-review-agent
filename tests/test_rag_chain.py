import pytest
from unittest.mock import MagicMock, patch

from review_assistant.rag_chain import RAGChain

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = MagicMock()
    mock.as_retriever.return_value = MagicMock()
    return mock

def test_rag_chain_initialization(mock_vector_store):
    """Test that RAGChain initializes correctly."""
    api_key = "test_api_key"
    chain = RAGChain(mock_vector_store, api_key)
    
    assert chain.vector_store == mock_vector_store
    assert chain.api_key == api_key
    assert chain.retriever is not None
    assert chain.prompt_template is not None
    assert chain.llm is None
    assert chain.chain is None

def test_create_prompt_template(mock_vector_store):
    """Test prompt template creation."""
    chain = RAGChain(mock_vector_store, "test_key")
    template = chain._create_prompt_template()
    
    assert template is not None
    # Check that the template contains expected text
    template_str = str(template.messages[0].prompt.template)
    assert "senior software engineer" in template_str
    assert "{context}" in template_str
    assert "{code}" in template_str

def test_build_chain(mock_vector_store):
    """Test chain building."""
    with patch("review_assistant.rag_chain.ChatOpenAI") as mock_openai:
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value="Review result")
        mock_openai.return_value = mock_llm
        
        chain = RAGChain(mock_vector_store, "test_key")
        built_chain = chain._build_chain()
        
        assert built_chain is not None
        assert chain.llm == mock_llm
        mock_openai.assert_called_once_with(model="gpt-4o", api_key="test_key")

def test_invoke(mock_vector_store):
    """Test invoking the RAG chain."""
    with patch("review_assistant.rag_chain.ChatOpenAI") as mock_openai:
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=MagicMock(content="This code looks good!"))
        mock_openai.return_value = mock_llm
        
        chain = RAGChain(mock_vector_store, "test_key")
        
        # Mock the chain's invoke method
        with patch.object(chain, '_build_chain') as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "This code looks good!"
            mock_build.return_value = mock_chain
            
            result = chain.invoke("def test(): pass")
            
            assert result == "This code looks good!"
            mock_chain.invoke.assert_called_once_with("def test(): pass")

def test_invoke_caches_chain(mock_vector_store):
    """Test that the chain is cached after first build."""
    with patch("review_assistant.rag_chain.ChatOpenAI"):
        chain = RAGChain(mock_vector_store, "test_key")
        
        with patch.object(chain, '_build_chain') as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Review"
            mock_build.return_value = mock_chain
            
            # First invocation
            chain.invoke("code1")
            assert mock_build.call_count == 1
            
            # Second invocation should not rebuild
            chain.invoke("code2")
            assert mock_build.call_count == 1  # Still 1, not rebuilt
