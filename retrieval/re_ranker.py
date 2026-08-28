import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .vector_search import get_chunks

model_name = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model = model.to("cuda")
model.eval()

collection_name = "embedd_test_collection"

def get_context(query, db_name:str = collection_name,top_k: int = 5):

    chunk_results = get_chunks(query,db_name)

    chunk_texts = [chunk[3] for chunk in chunk_results]

    pairs = [[query,txt] for txt in chunk_texts]

    
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to("cuda")

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            scores = model(**inputs).logits.view(-1).float()

    ranked = torch.argsort(scores,descending=True)[:top_k]

    rank_chunk_results = []

    for idx in ranked:
        idx_item = idx.item()
        original_chunk = chunk_results[idx_item]
        rank_chunk_results.append({
            "id": original_chunk[0],
            "chunk_id": original_chunk[1],
            "metadata": original_chunk[2],
            "text": original_chunk[3]
        })
        
    return rank_chunk_results










# def get_context(query, top_k: int = 5):

#     result_txt = get_chunks(query)

#     pairs = [[query,txt] for _,_,_,txt in result_txt]

#     with torch.no_grad():
#         inputs = tokenizer(
#             pairs,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to("cuda")

#         scores = model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

#     ranked = sorted(
#         zip(scores,result_txt),
#         key= lambda x:x[0],
#         reverse=True
#     )

#     chunk_results = []

#     for score,(id,chunk_id,meta,txt) in ranked[:top_k]:
#         #print(f"Score: {score}\n")
#         # print   (f"doc_name : {meta.get('doc_name')}\n")
#         # print(f"pg_no : {meta.get('page_no')}\n")
#         # print(txt)
#         # print("\n")
#         # print("="*60)
#         chunk_results.append([id,chunk_id,meta,txt])

#     context = ""

#     # for i,(txt,meta) in enumerate(chunk_results,start=1):
#     #     temp = f"Source {i}:\nContext: {txt}\nMetadata: {meta}\n\n---\n\n"
#     #     context += temp

#     # return context

#     return chunk_results

if __name__ == "__main__":

    query = """What is the council's Financial Management System used for, what areas did the 2015/16 audit focus on, and what issue did the auditors identify with the control accounts?"""
    chunk_results = get_context(query)

    for id,chunk_id,meta,txt in chunk_results:
        
        print(id,end="\n\n")
        print(chunk_id,end="\n\n")
        print(meta,end="\n\n")
        print(txt,end="\n\n")
        print("="*50)
        print("\n\n")





