import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from review_assistant.api import app, get_ingestor, get_rag_chain

# Create mock objects
mock_ingestor = MagicMock()
mock_rag_chain = MagicMock()

@pytest.fixture
def client():
    # Reset mocks before each test
    mock_ingestor.reset_mock()
    mock_rag_chain.reset_mock()
    # Override FastAPI dependencies to use our mocks
    app.dependency_overrides[get_ingestor] = lambda: mock_ingestor
    app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()

def test_upload_file(client):
    response = client.post(
        "/upload/",
        files={"file": ("test.py", b"def hello(): pass", "text/x-python")}
    )
    assert response.status_code == 201
    assert response.json() == {"message": "File 'test.py' uploaded and ingested successfully."}
    # The actual file path will be temporary, so we check that the method was called
    # without asserting the exact path.
    mock_ingestor.ingest_file.assert_called_once()

def test_review_code(client):
    mock_rag_chain.invoke.return_value = "This is a great function!"
    
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "def good_function():\\n    return True"

        response = client.post(
            "/review/",
            json={"file_path": "test_review_file.py"}
        )
        assert response.status_code == 200
        assert response.json() == {"review": "This is a great function!"}
        mock_rag_chain.invoke.assert_called_once_with("def good_function():\\n    return True")

def test_review_code_file_not_found(client):
    with patch("builtins.open", side_effect=FileNotFoundError):
        response = client.post(
            "/review/",
            json={"file_path": "non_existent_file.py"}
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "File not found."}