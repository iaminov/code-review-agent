import os
from pathlib import Path
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
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
        
        # Create a new index if it doesn't exist
        embedding_dimension = len(self.embeddings.embed_query("test"))
        dummy_index = faiss.IndexFlatL2(embedding_dimension)
        dummy_docstore = InMemoryDocstore({})
        
        return FAISS(
            embedding_function=self.embeddings,
            index=dummy_index,
            docstore=dummy_docstore,
            index_to_docstore_id={}
        )

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None):
        self.index.add_texts(texts=texts, metadatas=metadatas)

    def save_index(self):
        """Saves the FAISS index to the specified path."""
        self.index.save_local(str(self.index_path))

    def as_retriever(self):
        """
        Returns the vector store as a LangChain retriever.

        Returns:
            A LangChain retriever instance.
        """
        return self.index.as_retriever()
