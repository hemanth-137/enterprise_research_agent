import google.generativeai as genai
from chunking import process_doc_chunks
from doc_parser import doc_parser
from dotenv import load_dotenv
import os
import numpy as np
import uuid

load_dotenv()


file_path = "./data/pdfs/0070-pdf.pdf"
doc_doc = doc_parser(file_path)
processed_chunks = process_doc_chunks(doc_doc)
print("1. done doc_doc and processed chunks\n")

md_text = doc_doc.export_to_markdown()
with open("text_embedd1.txt","w",encoding="utf-8") as file:
    file.write(md_text)

print("done textembedd text 1\n")
with open("text_embedd2.txt","w",encoding="utf-8") as file:
    file.write(str(processed_chunks))
print("done textembedd text 2\n")
chunk_texts = [chunk["text"] for chunk in processed_chunks]



gemini_api = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api)

result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=chunk_texts,
        task_type="retrieval_document",
        output_dimensionality=768
        )

print("2. got embeddings\n")
embeddings_matrix = np.array(result['embedding'])
norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
normalized_embeddings = embeddings_matrix / norms

for i, chunk in enumerate(processed_chunks):
    chunk["embedding"] = normalized_embeddings[i].tolist()

with open("text_embedd3.txt","w",encoding="utf-8") as file:
    file.write(str(processed_chunks))

print("3. got full embeddings\n")
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


client = QdrantClient(url="http://localhost:6333")


client.create_collection(
    collection_name="embedd_test_collection",
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
print("4. added to db")














# with open("embed_vec.txt","w",encoding="utf-8") as file:

#         for i, chunk in enumerate(processed_chunks):
#                 file.write(chunk)






# print("==================result========================")
# print(result)

# embedding_values = result['embedding']
# embedding_length = len(embedding_values)

# print("==================embedding========================")
# print(f"Normal embedding length: {len(embedding_values)}")
# print(f"Norm of normal embedding: {np.linalg.norm(embedding_values):.6f}")
# print(f"normal embeddings: {embedding_values[:10]}")


# embedding_values_np = np.array(embedding_values)
# normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)

# print("==================embedding========================")
# print(f"Normed embedding length: {len(normed_embedding)}")
# print(f"Norm of normed embedding: {np.linalg.norm(normed_embedding):.6f}")
# print(f"norm embeddings: {normed_embedding[:10]}")

# print("==========================================================")
# print(f"Length of embedding: {embedding_length}")