import os
os.environ["HF_HUB_OFFLINE"] = "1"
from dotenv import load_dotenv
import json

from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from retrieval.re_ranker import get_context
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from .sys_prompts import answer_system_prompt,router_system_prompt

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query:str

class QueryAnalysis(BaseModel):
    needs_retrieval: bool = Field(
        description="False for greetings, small talk, or general conversational chat (e.g., 'hi', 'thanks', 'who are you?'). True for factual, domain-specific, or document queries."
    )
    optimized_query: str = Field(
        description="The query ready for vector search. Correct spelling. If vague, add minimal essential search keywords. If clear/detailed or small talk, keep it virtually unchanged."
    )


#print("Loading Llama 3.1...")

#subq_llm = ChatOllama(model="llama3.1", temperature=0.1, keep_alive=-1)
router_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite").with_structured_output(QueryAnalysis)
answer_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
print("llm loaded")



# subq_template = ChatPromptTemplate.from_messages([
#     ("system",subq_system_prompt),
# ("user","User Query: {query}")
# ])


answer_template = ChatPromptTemplate.from_messages([
    ("system", answer_system_prompt),
    ("human", """Here are the required sources:
{context}
---
User Question: {query}"""
    )
])


router_template = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("human", "{query}")
])

#subq_chain = subq_template | subq_llm
answer_chain = answer_template | answer_llm | StrOutputParser()
direct_answer_chain = answer_llm | StrOutputParser()
router_chain = router_template | router_llm


# print("Pre-loading Llama 3.1 into GPU memory (Warmup)...")
# answer_llm.invoke("Hi")
# print("LLM Loaded")

print("RAG System Ready!")


@app.post("/query")
def run_rag(user_input: QueryRequest):

    raw_query = user_input.query

    quer_check: QueryAnalysis = router_chain.invoke({"query": raw_query})

    def generate():
        if not quer_check.needs_retrieval:

            yield json.dumps({"type": "metadata", "sources": []}) + "\n"

            for chunk in direct_answer_chain.stream(raw_query):
                if chunk:
                    yield json.dumps({"type": "text", "content": chunk}) + "\n"
            return

        processed_query = quer_check.optimized_query

        context_result = get_context(processed_query)

        context_blocks = []
        metadata_payload = []

        for i, (id, chunk_id, meta, txt) in enumerate(context_result, start=1):
            context_blocks.append(f"Source [{i}]:\n{txt}")
            
            metadata_payload.append({
                "source_id": i,
                "db_id": id,
                "chunk_id": chunk_id,
                "doc_name": meta.get("doc_name"),
                "page_no": meta.get("page_no"),
                "headings": meta.get("headings"),
                "text": txt
            })

        context_str = "\n\n".join(context_blocks) 

        #subq_response = subq_chain.invoke({"query": query})
        #sub_queries = [line.strip() for line in subq_response.content.split("\n") if line.strip()]

        yield json.dumps({"type": "metadata", "sources": metadata_payload}) + "\n"


        for chunk in answer_chain.stream({"context": context_str, "query": processed_query}):
            if chunk:
                yield json.dumps({"type": "text", "content": chunk}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")










#     answer = answer_chain.invoke({
#         "context":context_str,
#         "query":processed_query})

#     return {"meta":metadata_payload,
#    #         "context":[get_context(q) for q in sub_queries],
#             "context":context_str,
#             "answer": answer.content
#             }