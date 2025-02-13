import os
from pathlib import Path
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class _EmbeddingAdapter(Embeddings):
    """Adapter to ensure FAISS receives a proper Embeddings object even when mocked."""
    def __init__(self, impl):
        self._impl = impl

    def embed_documents(self, texts):
        if hasattr(self._impl, "embed_documents"):
            return self._impl.embed_documents(texts)
        # Fallback: derive document embeddings from embed_query when necessary
        return [self._impl.embed_query(t) for t in texts]

    def embed_query(self, text):
        if hasattr(self._impl, "embed_query"):
            return self._impl.embed_query(text)
        raise NotImplementedError("embed_query not available on wrapped implementation")

class VectorStore:
    """A wrapper for FAISS vector store operations."""

    def __init__(self, index_path: str | os.PathLike):
        """
        Initializes the VectorStore.

        Args:
            index_path: The path to the FAISS index file.
        """
        self.index_path = Path(index_path)
        self.embeddings = OpenAIEmbeddings()
        self._embedding = _EmbeddingAdapter(self.embeddings)
        self.index = self._load_index()

    def _load_index(self) -> FAISS:
        """
        Loads the FAISS index from the specified path, or creates a new one.

        Returns:
            A FAISS vector store instance.
        """
        if self.index_path.exists():
            return FAISS.load_local(str(self.index_path), self._embedding, allow_dangerous_deserialization=True)

        embedding_dimension = len(self._embedding.embed_query("test"))
        dummy_index = faiss.IndexFlatL2(embedding_dimension)
        dummy_docstore = InMemoryDocstore({})

        return FAISS(
            embedding_function=self._embedding,
            index=dummy_index,
            docstore=dummy_docstore,
            index_to_docstore_id={},
        )

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None):
        """
        Adds texts to the vector store.

        Args:
            texts: A list of texts to add.
            metadatas: Optional list of metadata dictionaries corresponding to the texts.
        """
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