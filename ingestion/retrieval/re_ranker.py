import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from vector_search import create_query_embeddings, get_chunks


model_name = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model = model.to("cuda")
model.eval()


query = """What planning application and development are being discussed, 
what change did the commenter make from their 
previous position, and what local reason did they give for supporting approval?"""

query_embedd = create_query_embeddings(query)
result_txt = get_chunks(query_embedd)

pairs = [[query,txt] for _,txt in result_txt]

with torch.no_grad():
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to("cuda")

    scores = model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

ranked = sorted(
    zip(scores,result_txt),
    key= lambda x:x[0],
    reverse=True
)


for score,(meta,txt) in ranked[:5]:
    print(f"Score: {score}\n")
    print(f"doc_name : {meta.get('doc_name')}\n")
    print(f"pg_no : {meta.get('page_no')}\n")
    print(txt)
    print("\n")
    print("="*60)