from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from generation.test_app import run_rag, QueryRequest

app = FastAPI()

DB_NAME = "embedd_test_collection"

@app.post("/query")
def query_endpoint(request: QueryRequest):

    generator = run_rag(request, db_name=DB_NAME)

    return StreamingResponse(
        generator, 
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    #print("Starting on http://localhost:8000")
    pass