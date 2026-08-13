import google.generativeai as genai

from dotenv import load_dotenv
import os
import numpy as np
from numpy.linalg import norm



load_dotenv()

gemini_api = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api)

result = genai.embed_content(
        model="models/gemini-embedding-001",
        content="What is the meaning of life?",
        task_type="retrieval_document",
        output_dimensionality=768
        )



embedding_values = result['embedding']
embedding_length = len(embedding_values)

embedding_values_np = np.array(embedding_values)
normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)

print(f"Normed embedding length: {len(normed_embedding)}")
print(f"Norm of normed embedding: {np.linalg.norm(normed_embedding):.6f}")


print(embedding_values)
print(f"Length of embedding: {embedding_length}")