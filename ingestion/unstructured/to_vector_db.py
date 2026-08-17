from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


url = "http://localhost:6333"

def add_embedd_to_db(processed_chunks,db_name,client_url = url):


    client = QdrantClient(url=client_url)

    if not client.collection_exists(db_name):
        client.create_collection(
            collection_name=db_name,
            vectors_config=VectorParams(size=768, distance=Distance.DOT),
        )

    points = []

    for chunk in processed_chunks:
        qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]))

        payload = {
            "id": chunk["id"],   # Keeps your original readable ID
            "text": chunk["text"],         # Chunk content
            "metadata": chunk["metadata"]  # Custom metadata dict
        }

        points.append(
            PointStruct(
                id=qdrant_id,
                vector=chunk["embedding"],
                payload=payload
            )
        )

    operation_info = client.upsert(
        collection_name="embedd_test_collection",
        points=points,
        wait=True
    )

    return operation_info