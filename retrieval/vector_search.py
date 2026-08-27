import os
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient


model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)


url = "http://localhost:6333"
client = QdrantClient(url=url)
collection_name = "open_ragbench__collection"

def create_query_embeddings(query):

    instruction = "Represent this sentence for searching relevant passages: "
    embedding = model.encode(
        instruction+query,
        normalize_embeddings=True
    ).tolist()

    return embedding


def get_chunks(embedding):

    results = client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=30
    )

    ret_txt = []
    
    for result in results.points:

        id = result.id
        chunk_id = result.payload['id']
        meta = result.payload['metadata']
        txt = result.payload['text']

        # print(f"\nScore: {result.score}\n")
        # print(f"doc_name : {meta.get('doc_name')}\n")
        # print(f"pg_no : {meta.get('page_no')}\n")
        # print(txt)
        # print("="*60)
        ret_txt.append([id,chunk_id,meta,txt])

    return ret_txt

if __name__ == "__main__":

    query = """What is the council's Financial Management System used for, what areas did the 2015/16 audit focus on, and what issue did the auditors identify with the control accounts?"""

    embedding = create_query_embeddings(query)
    results = get_chunks(embedding)
    for id,chunk_id,meta,txt in results[:3]:
        print(id,end="\n\n")
        print(chunk_id,end="\n\n")
        print(meta,end="\n\n")
        print(txt,end="\n\n")
        print("="*50)
        print("\n\n")