import os
from pathlib import Path
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStore:
    """A wrapper for FAISS vector store operations."""

    def __init__(self, index_path: str | os.PathLike):
        self.index_path = Path(index_path)
        self.embeddings = OpenAIEmbeddings()
        self.index = self._load_index()

    def _load_index(self) -> FAISS:
        if self.index_path.exists():
            return FAISS.load_local(str(self.index_path), self.embeddings)
        
        # BUG: Not creating a new index properly
        return None

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None):
        self.index.add_texts(texts=texts, metadatas=metadatas)

    def save_index(self):
        self.index.save_local(str(self.index_path))
