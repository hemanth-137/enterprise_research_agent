from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)

def create_embeddings(chunks_generator, batch_size=32):
    batch = []
    
    for chunk in chunks_generator:
        batch.append(chunk)
        
        if len(batch) == batch_size:
            texts = [item["text"] for item in batch]
            
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            for item, embedding in zip(batch, embeddings):
                item["embedding"] = embedding.tolist()
                yield item
                
            batch = []

    if batch:
        texts = [item["text"] for item in batch]
        embeddings = model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            show_progress_bar=False
        )
        for item, embedding in zip(batch, embeddings):
            item["embedding"] = embedding.tolist()
            yield item