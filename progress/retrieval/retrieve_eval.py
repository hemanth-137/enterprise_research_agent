import torch
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from retrieval.vector_search import create_query_embeddings
from retrieval.eval_set import evaluation_set

# ---------------- MODEL ----------------

model_name = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model = model.to("cuda")
model.eval()


# ---------------- QDRANT ----------------

client = QdrantClient(url="http://localhost:6333")

collection_name = "embedd_test_collection"


# ---------------- EVALUATION ----------------

for item in evaluation_set:

    query = item["question"]
    target_chunk_id = item["chunk_id"]
    target_doc = item["doc_name"]

    # Query embedding
    query_embedding = create_query_embeddings(query)

    # Direct Qdrant search -- preserve point ID
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=30
    )

    # Store metadata + text + ID
    result_txt = []

    for result in results.points:

        meta = result.payload["metadata"]
        txt = result.payload["text"]

        result_txt.append({
            "chunk_id": result.id,
            "metadata": meta,
            "text": txt,
            "qdrant_score": result.score
        })

    # ---------------- QDRANT RANK ----------------

    qdrant_found = False

    for rank, result in enumerate(result_txt, start=1):

        if str(result["chunk_id"]) == target_chunk_id:

            print("\n================ QDRANT ================")
            print(f"Question    : {query}")
            print(f"Target ID   : {target_chunk_id}")
            print(f"Qdrant rank : {rank}")
            print(f"Doc name    : {result['metadata'].get('doc_name')}")

            qdrant_found = True
            break

    if not qdrant_found:

        print("\n================ QDRANT ================")
        print(f"Question    : {query}")
        print(f"Target ID   : {target_chunk_id}")
        print("NOT FOUND IN TOP-30")

    # ---------------- RERANK ----------------

    pairs = [
        [query, result["text"]]
        for result in result_txt
    ]

    with torch.no_grad():

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to("cuda")

        scores = (
            model(**inputs, return_dict=True)
            .logits
            .view(-1)
            .float()
            .cpu()
            .tolist()
        )

    ranked = sorted(
        zip(scores, result_txt),
        key=lambda x: x[0],
        reverse=True
    )

    # ---------------- RERANK RANK ----------------

    rerank_found = False

    for rank, (score, result) in enumerate(ranked, start=1):

        if str(result["chunk_id"]) == target_chunk_id:

            print("\n================ RERANKER ================")
            print(f"Target ID    : {target_chunk_id}")
            print(f"Rerank rank  : {rank}")
            print(f"Rerank score : {score}")
            print(f"Doc name     : {result['metadata'].get('doc_name')}")

            rerank_found = True
            break

    if not rerank_found:

        print("\n================ RERANKER ================")
        print(f"Target ID : {target_chunk_id}")
        print("NOT FOUND")


    print("*"*30)