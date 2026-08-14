from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()


client = QdrantClient(url="http://localhost:6333")

gemini_api = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api)

query = """
When does the call for evidence for Approved Document B conclude?"""



result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="question_answering",
        output_dimensionality=768
        )

embedding_vector = np.array(result['embedding'])
norm = np.linalg.norm(embedding_vector)
normalized_embeddings = (embedding_vector / norm).tolist()

search_result = client.query_points(
    collection_name="embedd_test_collection",
    query=normalized_embeddings,
    with_payload=True,
    limit=2
).points

for i in search_result:

    
    print(f"Qdrant ID (UUID): {i.id}")
    print(f"Match Score: {i.score:.4f}")
    print("=" * 30)
    
    # 2. Use bracket notation to read keys inside your payload dictionary
    print(f"Original Chunk ID: {i.payload['id']}")
    print(f"Text Content:\n{i.payload['text']}")
    print("\n" + "#" * 50 + "\n")