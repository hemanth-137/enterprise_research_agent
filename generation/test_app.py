import os
os.environ["HF_HUB_OFFLINE"] = "1"

from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from retrieval.re_ranker import get_context
from pydantic import BaseModel
from fastapi.responses import StreamingResponse


app = FastAPI()

class QueryRequest(BaseModel):
    query:str


print("Loading Llama 3.1...")

subq_llm = ChatOllama(model="llama3.1", temperature=0.1, keep_alive=-1)
answer_llm = ChatOllama(model="llama3.1", temperature=0.5, keep_alive=-1)

subq_system_prompt = """You are a query expansion model used in a Retrieval-Augmented Generation (RAG) system.

Your task is to transform a user's query into a small set of independent retrieval sub-queries that can be searched against a document collection.

The goal is to maximize retrieval recall while keeping the queries focused and relevant.

Instructions
- Identify the distinct pieces of information the user is asking for.
- Break complex or multi-part questions into separate, self-contained sub-queries.
- Each sub-query should represent one meaningful information need.
- Make each sub-query understandable on its own without requiring the original query.
- Preserve important entities, names, products, organizations, dates, technical terms, and constraints from the original query.
- When useful, rephrase the query using terminology that is likely to appear in source documents.
- Do not invent facts, entities, names, dates, or terminology that are not supported by the original query.
- Do not answer the user's question.
- Do not include explanations, reasoning, or commentary.
- Avoid redundant sub-queries. Generate only the queries that provide distinct retrieval value.
- For simple queries, return one sub-query instead of unnecessarily splitting the query.
- For complex queries, generate between 2 and 6 sub-queries.

Important
- The sub-queries will be used for document retrieval.
- Prefer queries that are likely to match useful source chunks rather than queries that merely sound conversational.

For example, instead of:

"Can you tell me what projects Jason worked on?"

prefer:

"Projects completed by Jason in artificial intelligence and machine learning"

Output Format
- Output ONLY the raw sub-queries, one per line.
- Do NOT use bullet points, numbering, or dashes.
- Do NOT use any quotation marks (no " or ').
- Do NOT output any JSON, brackets, or code blocks.

Correct Output Example:
First sub query
Second sub query"""


subq_template = ChatPromptTemplate.from_messages([
    ("system",subq_system_prompt),
("user","User Query: {query}")
])

answer_system_prompt = """You are a precise and strictly grounded QA assistant. Your task is to answer the user's question based ONLY on the provided Sources below.

To remain trustworthy, you must adhere to these absolute rules:

1. STRICT GROUNDING: Answer the question using ONLY the facts explicitly mentioned in the provided sources. Do NOT use your own pre-trained knowledge, do NOT extrapolate, and do NOT make assumptions.
2. UNANSWERABLE FALLBACK: If the provided sources do not contain enough information to fully answer the question, reply exactly with: "I'm sorry, but I do not have enough information in the provided documents to answer that." Do not attempt to draft a partial or guess answer.
3. EXPLICIT CITATIONS: For every single claim or fact you state in your response, you MUST cite the source from which it was extracted using provided metadata. Use the format "[Source '{{doc name, page. no}}']" (e.g., "[Source name.pdf, pg.no: 2]") at the end of the sentence or clause.
4. TONE: Be direct, clear, and professional. Do not say "Based on Source 1..." or "The provided text states...". Just answer the query directly with citations embedded."""

answer_template = ChatPromptTemplate.from_messages([
    ("system", answer_system_prompt),
    ("human", """Here are the retrieved sources:
{context}
---
User Question: {query}"""
    )
])


subq_chain = subq_template | subq_llm
answer_chain = answer_template | answer_llm


print("Pre-loading Llama 3.1 into GPU memory (Warmup)...")
answer_llm.invoke("Hi")
print("LLM Loaded")

print("RAG System Ready!")


@app.post("/query")
def run_rag(user_input: QueryRequest):

    query = user_input.query
    context = get_context(query)

    #subq_response = subq_chain.invoke({"query": query})
    #sub_queries = [line.strip() for line in subq_response.content.split("\n") if line.strip()]

    def generate():
        for chunk in answer_chain.stream({
            "context": context,
            "query": query
        }):
            yield chunk.content
#     answer = answer_chain.invoke({"context":context,"query":query})

#     return {"subq":answer.content,
#    #         "context":[get_context(q) for q in sub_queries],
#             "context":context,
#             "answer": answer.content
#             }
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )
