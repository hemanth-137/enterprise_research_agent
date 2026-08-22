import os
os.environ["HF_HUB_OFFLINE"] = "1"

from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from retrieval.re_ranker import get_context

app = FastAPI()

print("Loading Llama 3.1...")
llm = ChatOllama(model="llama3.1", temperature=0.8, keep_alive=-1)

template = ChatPromptTemplate.from_messages([
    ("system", """You are a precise and strictly grounded QA assistant. Your task is to answer the user's question based ONLY on the provided Sources below.

To remain trustworthy, you must adhere to these absolute rules:

1. STRICT GROUNDING: Answer the question using ONLY the facts explicitly mentioned in the provided sources. Do NOT use your own pre-trained knowledge, do NOT extrapolate, and do NOT make assumptions.
2. UNANSWERABLE FALLBACK: If the provided sources do not contain enough information to fully answer the question, reply exactly with: "I'm sorry, but I do not have enough information in the provided documents to answer that." Do not attempt to draft a partial or guess answer.
3. EXPLICIT CITATIONS: For every single claim or fact you state in your response, you MUST cite the source from which it was extracted using provided metadata. Use the format "[Source '{{doc name, page. no}}']" (e.g., "[Source name.pdf, pg.no: 2]") at the end of the sentence or clause.
4. TONE: Be direct, clear, and professional. Do not say "Based on Source 1..." or "The provided text states...". Just answer the query directly with citations embedded."""
     ),

    ("human", """Here are the retrieved sources:
{context}
---
User Question: {query}"""
     )
])

chain = template | llm


print("Pre-loading Llama 3.1 into GPU memory (Warmup)...")
llm.invoke("Hi")

print("RAG System Ready!")



@app.post("/query")
def run_rag(user_input: dict):
    query = user_input["query"]
    
    context = get_context(query)
    
    response = chain.invoke({"context": context, "query": query})
    
    return {"context":context,
            "answer": response.content
            }