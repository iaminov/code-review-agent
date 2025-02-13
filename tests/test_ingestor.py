import pytest
from unittest.mock import MagicMock
from review_assistant.ingestor import Ingestor

@pytest.fixture
def mock_vector_store():
    return MagicMock()

@pytest.fixture
def ingestor(mock_vector_store):
    return Ingestor(vector_store=mock_vector_store)

def test_ingest_file(ingestor, tmp_path):
    file_path = tmp_path / "test_file.py"
    file_path.write_text("def hello():\n    print('world')")
    ingestor.ingest_file(file_path)
    ingestor.vector_store.add_texts.assert_called_once()

def test_ingest_directory(ingestor, tmp_path):
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    file1 = dir_path / "file1.py"
    file1.write_text("import os")
    file2 = dir_path / "file2.js"
    file2.write_text("console.log('hello');")

    ingestor.ingest_directory(dir_path)
    assert ingestor.vector_store.add_texts.call_count == 2
    ingestor.vector_store.save_index.assert_called_once()