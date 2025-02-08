import os
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from review_assistant.vector_store import VectorStore
from review_assistant.ingestor import Ingestor
from review_assistant.rag_chain import RAGChain

# Load API key from environment variable lazily (do not raise at import-time)
def _get_api_key() -> str | None:
    # Read lazily so tests can import and override dependencies without needing the env var set
    return os.getenv("OPENAI_API_KEY")

# Use lru_cache to ensure components are created only once
@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(index_path="vector_store.faiss")

@lru_cache
def get_ingestor() -> Ingestor:
    return Ingestor(vector_store=get_vector_store())

@lru_cache
def get_rag_chain() -> RAGChain:
    api_key = _get_api_key()
    return RAGChain(vector_store=get_vector_store(), api_key=api_key)

app = FastAPI(
    title="Code Review Assistant",
    description="An autonomous code review assistant using RAG.",
)

class ReviewRequest(BaseModel):
    file_path: str
    focus: str | None = None

class ReviewResponse(BaseModel):
    review: str

@app.post("/upload/", status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    ingestor: Ingestor = Depends(get_ingestor)
):
    """
    Uploads a file for ingestion into the vector store.
    """
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Ingest the file
        ingestor.ingest_file(temp_file_path)

        # Clean up the temporary file
        os.remove(temp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"File '{file.filename}' uploaded and ingested successfully."}

@app.post("/review/", response_model=ReviewResponse)
async def review_code(
    request: ReviewRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
):
    """
    Reviews a code file and returns feedback.
    """
    try:
        with open(request.file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    review = rag_chain.invoke(code)
    return ReviewResponse(review=review)
