import os
from pathlib import Path
from review_assistant.vector_store import VectorStore

class Ingestor:
    """Handles the ingestion of code files into the vector store."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        # TODO: Add text splitting

    def ingest_file(self, file_path: str | os.PathLike):
        path = Path(file_path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            print(f"Error reading file {path}: {e}")
            return

        # BUG: Not chunking text properly
        self.vector_store.add_texts([content])
