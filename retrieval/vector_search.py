import os
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient


model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cpu"
)


url = "http://localhost:6333"
client = QdrantClient(url=url)


def create_query_embeddings(query):


    embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    return embedding


def get_chunks(embedding):

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

    query = """What is the council's Financial Management System used for, what areas did the 2015/16 audit focus on, and what issue did the auditors identify with the control accounts?"""

    embedding = create_query_embeddings(query)
    results = get_chunks(embedding)
    for meta,txt in results:
        print(meta)
        #print("\n\n")
        print()
        print(txt)
        print()
        print("="*50)
        print("\n\n")