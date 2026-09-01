from retrieval.re_ranker import get_context
from evaluation.open_ragbench_eval import get_queries
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import json

answer_system_prompt = """
You are a question-answering assistant. Answer the user's question using only the provided context.

- Give a direct, concise, self-contained answer.
- Synthesize the relevant information from the context rather than copying large passages.
- Include the key facts needed to answer the question accurately.
- Match the specificity of the question; do not add unnecessary background.
- Do not use information that is not supported by the context.
- Do not mention the context, sources, retrieval process, or these instructions.
- If the context does not contain enough information to answer, say: "I couldn't find relevant information in the provided context to answer this."
"""

answer_llm = ChatOllama(model="llama3.1",temperature = 0.2)
answer_template = ChatPromptTemplate.from_messages([
    ("system", answer_system_prompt),
    ("human", """Retrieved context:
{context}
---
User Question: {query}"""
    )
])
answer_chain = answer_template | answer_llm | StrOutputParser()

print("LLM loaded...\n\n")


def retrieve_context(eval_query,db_name: str = "open_ragbench__collection",limit: int = 3):

    for query_eval in eval_query:
        query = query_eval["query"]
        chunks = get_context(query,db_name,limit)

        context_blocks = []

        for i, chunk in enumerate(chunks, start=1):
            context_blocks.append(f"Source [{i}]:\n{chunk['text']}")

        context_str = "\n\n".join(context_blocks)

        query_eval["context"] = context_str

    return eval_query



def to_llm(eval_query):
    temp = 0
    for query_set in eval_query:
        temp+=1
        answer = answer_chain.invoke({"context":query_set["context"],"query":query_set["query"]})
        query_set["generated_answer"] = answer
        if temp%5==0:
            print(f"total llm generations done = {temp}\n\n")

    return eval_query

eval_set = get_queries(n = 100)
eval_set = retrieve_context(eval_set)
eval_set = to_llm(eval_set)


# for i in eval_set:
#     print(f"Query: {i['query']}\n")
#     print(f"Original Answer: {i['answer']}\n")
#     print(f"Generated Answer: {i['generated_answer']}\n")
#     print(f"Provided Context: {i['context']}\n")
#     print(f"Type: {i['type']}\n")
#     print(f"Source: {i['source']}\n")
#     print("="*20,end="\n")



with open("rag_eval_generate_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_set, f, indent=2, ensure_ascii=False)

print(f"Saved {len(eval_set)} evaluation results to rag_eval_results.json")



























# def get_answer(query_results,db_name: str = "open_ragbench__collection",limit: int = 3):
#     total = 0
#     true = 0
#     false = 0
#     for query,file_name in query_results:
#         got_it = False
#         total += 1
#         chunks = get_context(query,db_name,limit)
#         for chunk in chunks:
#             curr_name = chunk["metadata"]["doc_name"]
#             # print(f"Original: {file_name}\nResult_name: {curr_name}")
#             # print("="*20)
#             if curr_name == file_name:
#                 got_it = True
#                 break

#         if got_it:
#             true+=1
#         else:
#             false+=1

#         if total%100==0:
#             print(f"completed {total} queries\n")

#     print(f"Total: {total}\nTrue: {true}\nFalse: {false}\n\n")
#     print(f"Recall@{limit}: {true / total:.4f}")


# queries = get_queries()
# get_answer(query_results=queries)


# Loading weights: 100%|█| 199/199 [00:00<00:00
# Loading weights: 100%|█| 201/201 [00:00<00:00
# completed 100 queries

# completed 200 queries

# completed 300 queries

# completed 400 queries

# completed 500 queries

# completed 600 queries

# completed 700 queries

# completed 800 queries

# completed 900 queries

# completed 1000 queries

# Total: 1000
# True: 979
# False: 21


# Recall@3: 0.9790