import os
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)


url = "http://localhost:6333"
client = QdrantClient(url=url)
collection_name = "embedd_test_collection"

def create_query_embeddings(query):

    instruction = "Represent this sentence for searching relevant passages: "
    embedding = model.encode(
        instruction+query,
        normalize_embeddings=True
    ).tolist()

    return embedding


def get_chunks(query,db_name:str = collection_name,limit: int = 30):

    embedding = create_query_embeddings(query)

    results = client.query_points(
        collection_name=db_name,
        query=embedding,
        limit=limit
    ).points

    ret_txt = []
    
    for result in results:

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