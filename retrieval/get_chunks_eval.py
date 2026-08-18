from qdrant_client import QdrantClient


url = "http://localhost:6333"
collection_name = "embedd_test_collection"

client = QdrantClient(url=url)

output_file = "qdrant_chunks.txt"

with open(output_file, "w", encoding="utf-8") as f:

    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in records:

            payload = point.payload

            chunk_id = point.id
            text = payload.get("text", "")
            metadata = payload.get("metadata", {})

            doc_name = metadata.get("doc_name")
            page_no = metadata.get("page_no")

            f.write("=" * 100 + "\n")
            f.write(f"CHUNK ID: {chunk_id}\n")
            f.write(f"DOC NAME: {doc_name}\n")
            f.write(f"PAGE NO: {page_no}\n")
            f.write("TEXT:\n")
            f.write(text)
            f.write("\n\n")

        if offset is None:
            break

print(f"Finished. Chunks saved to: {output_file}")