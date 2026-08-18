from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient


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
        limit=30
    )

    ret_txt = []
    
    for result in results.points:

        meta = result.payload['metadata']
        txt = result.payload['text']

        # print(f"\nScore: {result.score}\n")
        # print(f"doc_name : {meta.get('doc_name')}\n")
        # print(f"pg_no : {meta.get('page_no')}\n")
        # print(txt)
        # print("="*60)
        ret_txt.append([meta,txt])

    return ret_txt

if __name__ == "__main__":

    query = """What planning application and development are 
    being discussed, what change did the commenter make from their
      previous position, and what local reason did they give for supporting approval?"""

    embedding = create_query_embeddings(query)
    get_chunks(embedding)
