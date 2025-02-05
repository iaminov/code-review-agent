import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from review_assistant.vector_store import VectorStore

class Ingestor:
    """Handles the ingestion of code files into the vector store."""

    def __init__(self, vector_store: VectorStore):
        """
        Initializes the Ingestor.

        Args:
            vector_store: An instance of the VectorStore.
        """
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

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
