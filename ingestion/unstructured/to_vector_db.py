import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

url = "http://localhost:6333"

def add_embedd_to_db_stream(embedded_chunks_generator, db_name, client_url=url, batch_size=500):

    client = QdrantClient(url=client_url)

    if not client.collection_exists(db_name):
        client.create_collection(
            collection_name=db_name,
            vectors_config=VectorParams(size=768, distance=Distance.DOT), # embedds are already normalized no need for cosine
        )

    batch_points = []
    total_upserted = 0

    for chunk in embedded_chunks_generator:
        qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]))

        payload = {
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        }

        point = PointStruct(
            id=qdrant_id,
            vector=chunk["embedding"],
            payload=payload
        )
        batch_points.append(point)

        if len(batch_points) == batch_size:  #for every batch size push to DB
            client.upsert(
                collection_name=db_name,
                points=batch_points,
                wait=True
            )
            total_upserted += len(batch_points)
            print(f"Upserted batch of {len(batch_points)} points to db. (Total: {total_upserted})")
            batch_points = []

    if batch_points:
        client.upsert(
            collection_name=db_name,
            points=batch_points,
            wait=True
        )
        total_upserted += len(batch_points)
        print(f"Upserted final batch of {len(batch_points)} points. Ingestion complete! (Total: {total_upserted})")

    return total_upserted