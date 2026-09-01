import os
from dotenv import load_dotenv
import json
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from retrieval.re_ranker import get_context
from .sys_prompts import answer_system_prompt, router_system_prompt, subq_system_prompt

os.environ["HF_HUB_OFFLINE"] = "1"

load_dotenv()

class ChatMessage(BaseModel):
    role: str
    content: str
class QueryRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = Field(default_factory=list)
class SubQueries(BaseModel):
    queries: List[str] = Field(
        description="A list of up to 3 distinct, concise, single-topic search queries designed to retrieve missing information needed to answer the original complex question."
    )
class QueryAnalysis(BaseModel):
    needs_retrieval: bool = Field(
        description="False for greetings, small talk, or general conversational chat (e.g., 'hi', 'thanks', 'who are you?'). True for factual, domain-specific, or document queries."
    )
    need_subq: bool = Field(
        description="True if the query is complex, comparative, covers multiple distinct topics, or requires pulling facts from different documents/sections. False for straightforward, single-fact queries."
    )
    optimized_query: str = Field(
        description="Concise vector-search query. Remove filler and answer-style instructions, preserve the actual information need and important details, and correct obvious spelling errors."
        # description="The query ready for vector search. Correct spelling. If vague, add minimal essential search keywords. If clear/detailed or small talk, keep it virtually unchanged."
    )


router_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite").with_structured_output(QueryAnalysis)
subq_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite").with_structured_output(SubQueries)
answer_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

answer_template = ChatPromptTemplate.from_messages([
    ("system", answer_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", """Here are the required sources:
{context}
---
User Question: {query}"""
    )
])

subq_template = ChatPromptTemplate.from_messages([
    ("system", subq_system_prompt),
    ("user", "User Query: {query}")
])

router_template = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{query}")
])

direct_answer_template = ChatPromptTemplate.from_messages([
    ("system", answer_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{query}")
])


answer_chain = answer_template | answer_llm | StrOutputParser()
direct_answer_chain = direct_answer_template | answer_llm | StrOutputParser()
router_chain = router_template | router_llm
subq_chain = subq_template | subq_llm

def run_rag(user_input: QueryRequest,db_name: str = "open_ragbench__collection",limit: int = 5):

    raw_query = user_input.query
    recent_history = []

    for message in user_input.chat_history[-6:]:
        if message.role == "user":
            recent_history.append(
                HumanMessage(content=message.content)
            )
        elif message.role == "assistant":
            recent_history.append(
                AIMessage(content=message.content)
            )

    print("=" * 20)
    print("\nchat history: ")
    print(recent_history, end="\n")
    print("=" * 20)
    print("\n\n")

    def generate():

        quer_check: QueryAnalysis = router_chain.invoke({
            "query": raw_query,
            "chat_history": recent_history
        })

        if not quer_check.needs_retrieval:
            print(
                f"--> The query: {raw_query}\n"
                "It doesn't need any retreival\n\n"
            )
            yield json.dumps({"type": "metadata","sources": []}) + "\n"

            for chunk in direct_answer_chain.stream({
                "query": raw_query,
                "chat_history": recent_history,
            }):
                if chunk:
                    yield json.dumps({"type": "text","content": chunk}) + "\n"

            print("So Closing now...\n\n")
            return

        processed_query = quer_check.optimized_query

        print(processed_query)

        print(
            f"--> The query: {raw_query}\n"
            "It does need retreival\n\n"
        )

        context_result = []

        if quer_check.need_subq:

            print("\t- It also need subq")

            subq_result: SubQueries = subq_chain.invoke({
                "query": raw_query
            })

            subq_lst = subq_result.queries[:3]

            subq_chunks = []
            subq_ids = set()

            print("Here are the subq's generated: \n")
            print(subq_result)

            for i in subq_lst:
                temp = get_context(i,db_name,top_k=3)
                for chunk in temp:
                    chunk_id_val = chunk["id"]
                    if chunk_id_val not in subq_ids:
                        subq_ids.add(chunk_id_val)
                        subq_chunks.append(chunk)

            context_result = subq_chunks

        else:
            print("\t- It dont need subq")
            context_result = get_context(processed_query,db_name,top_k=limit)

        context_blocks = []
        metadata_payload = []

        for i, chunk in enumerate(context_result, start=1):

            context_blocks.append(
                f"Source [{i}]:\n{chunk['text']}"
            )

            metadata_payload.append({
                "source_id": i,
                "db_id": chunk["id"],
                "chunk_id": chunk["chunk_id"],
                "doc_name": chunk["metadata"].get("doc_name"),
                "page_no": chunk["metadata"].get("page_no"),
                "headings": chunk["metadata"].get("headings"),
                "text": chunk["text"]
            })

        context_str = "\n\n".join(context_blocks)

        yield json.dumps({"type": "metadata","sources": metadata_payload}) + "\n"

        for chunk in answer_chain.stream({
            "context": context_str,
            "query": raw_query,
            "chat_history": recent_history
        }):
            if chunk:
                yield json.dumps({"type": "text","content": chunk}) + "\n"

    return generate()