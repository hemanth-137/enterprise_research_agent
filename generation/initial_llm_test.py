from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# from ..retrieval.re_ranker import get_context


query = """what projects does the candiate hemanth varma got? I am looking to hire a gen ai candidate"""

context = get_context(query)

print(context)


# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0.7
# )

# template = ChatPromptTemplate.from_messages([
#     ("system", """You are Reed Richards
#     (Mr. Fantastic) from Marvel Comics Universe,
#     so answer as you are him."""),
#     ("human", "{user_input}")
# ])

# message = {
#     "user_input": "Who is Dr. Doom?"
# }

# chain = template | llm

# response = chain.invoke(message)

# print(response.content)