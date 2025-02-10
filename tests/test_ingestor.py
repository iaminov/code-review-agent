import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from review_assistant.ingestor import Ingestor

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = MagicMock()
    return mock

def test_ingestor_initialization(mock_vector_store):
    """Test that Ingestor initializes correctly."""
    ingestor = Ingestor(mock_vector_store)
    assert ingestor.vector_store == mock_vector_store
    assert ingestor.text_splitter is not None
    assert ingestor.text_splitter.chunk_size == 1000
    assert ingestor.text_splitter.chunk_overlap == 200

def test_ingest_file_success(mock_vector_store):
    """Test successful file ingestion."""
    ingestor = Ingestor(mock_vector_store)
    
    file_content = "def hello():\n    return 'world'"
    
    with patch("builtins.open", mock_open(read_data=file_content)):
        with patch.object(ingestor.text_splitter, 'create_documents') as mock_create_docs:
            mock_doc = MagicMock()
            mock_doc.page_content = file_content
            mock_doc.metadata = {}
            mock_create_docs.return_value = [mock_doc]
            
            ingestor.ingest_file("test.py")
            
            mock_create_docs.assert_called_once_with([file_content])
            mock_vector_store.add_texts.assert_called_once()

def test_ingest_file_io_error(mock_vector_store, capsys):
    """Test file ingestion with IO error."""
    ingestor = Ingestor(mock_vector_store)
    
    with patch("builtins.open", side_effect=IOError("Cannot read file")):
        ingestor.ingest_file("nonexistent.py")
        
        captured = capsys.readouterr()
        assert "Error reading file" in captured.out
        mock_vector_store.add_texts.assert_not_called()

def test_ingest_directory(mock_vector_store):
    """Test directory ingestion."""
    ingestor = Ingestor(mock_vector_store)
    
    with patch.object(Path, 'rglob') as mock_rglob:
        mock_file1 = MagicMock(spec=Path)
        mock_file1.is_file.return_value = True
        mock_file2 = MagicMock(spec=Path)
        mock_file2.is_file.return_value = True
        mock_dir = MagicMock(spec=Path)
        mock_dir.is_file.return_value = False
        
        mock_rglob.return_value = [mock_file1, mock_file2, mock_dir]
        
        with patch.object(ingestor, 'ingest_file') as mock_ingest:
            ingestor.ingest_directory("test_dir")
            
            assert mock_ingest.call_count == 2
            mock_vector_store.save_index.assert_called_once()
