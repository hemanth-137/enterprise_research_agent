from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)



def create_embeddings(processed_chunks, batch_size=32):

    texts = [chunk["text"] for chunk in processed_chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    for chunk, embedding in zip(processed_chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return processed_chunks