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
        """
        Reads, chunks, and embeds a single file.

        Args:
            file_path: The path to the file to ingest.
        """
        path = Path(file_path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            print(f"Error reading file {path}: {e}")
            return

        documents = self.text_splitter.create_documents([content])
        for doc in documents:
            doc.metadata["source"] = str(path.as_posix())

        self.vector_store.add_texts(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

    def ingest_directory(self, directory_path: str | os.PathLike):
        """
        Recursively ingests all supported file types in a directory.

        Args:
            directory_path: The path to the directory to ingest.
        """
        path = Path(directory_path)
        for item in path.rglob("*"):
            if item.is_file():
                self.ingest_file(item)

        self.vector_store.save_index()
