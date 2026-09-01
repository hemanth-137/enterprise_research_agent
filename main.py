from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import List
from pydantic import BaseModel, Field

from generation.generate_answer import run_rag

app = FastAPI()

DB_NAME = "embedd_test_collection"


class ChatMessage(BaseModel):
    role: str
    content: str
class QueryRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = Field(default_factory=list)


@app.post("/query")
def query_endpoint(request: QueryRequest):

    generator = run_rag(
        request,
        db_name=DB_NAME
    )

    return StreamingResponse(
        generator,
        media_type="application/x-ndjson"
    )


if __name__ == "__main__":
    #print("Starting on http://localhost:8000")
    pass