from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)


url = "http://localhost:6333"



def create_query_embeddings(query):


    embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    return embedding

def get_chunks(embedding):

    client = QdrantClient(url=url)

    results = client.query_points(
        collection_name="embedd_test_collection",
        query=embedding,
        limit=5
    )

    for result in results.points:
        print(f"\nScore: {result.score}\n")
        print(f"doc_name : {result.payload['metadata'].get('doc_name')}\n")
        print(f"pg_no : {result.payload['metadata'].get('page_no')}\n")
        print(result.payload["text"])
        print("="*60)


query = """What adjustment was made to account for differences between direct and indirect survey responses?"""


embedding = create_query_embeddings(query)
get_chunks(embedding)
