from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Code Review Assistant",
    description="An autonomous code review assistant using RAG."
)

class ReviewRequest(BaseModel):
    code: str

@app.get("/")
def read_root():
    return {"message": "Code Review Assistant API"}

@app.post("/review")
def review_code(request: ReviewRequest):
    # TODO: Implement actual review logic
    return {"review": "Code review not yet implemented"}
