import streamlit as st
import requests
import json

st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Local RAG Assistant")

BACKEND_URL = "http://localhost:8000/query"

# Initialize Chat History in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper function to render sources expander
def render_sources_expander(sources):
    if not sources:
        return
    with st.expander("🔍 View Retrieved Context & Sources"):
        for src in sources:
            page_str = ", ".join(map(str, src['page_no'])) if isinstance(src['page_no'], list) else src['page_no']
            headings_str = " > ".join(src['headings']) if src['headings'] else "None"
            
            st.markdown(f"### Source [{src['source_id']}]: `{src['doc_name']}` (Page: {page_str})")
            st.caption(f"**DB ID:** `{src['db_id']}` | **Chunk ID:** `{src['chunk_id']}` | **Section:** {headings_str}")
            st.code(src['text'], language="text")
            st.divider()

# 1. Render existing chat history on app reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources_expander(message["sources"])

# 2. Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare assistant response container
    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        full_response = ""
        retrieved_sources = []

        try:
            # Stream from FastAPI endpoint
            with requests.post(
                BACKEND_URL,
                json={"query": prompt},
                stream=True
            ) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if line:
                        event = json.loads(line.decode("utf-8"))
                        
                        # Event 1: Extract Metadata
                        if event["type"] == "metadata":
                            retrieved_sources = event["sources"]
                        
                        # Event 2: Stream Text Tokens
                        elif event["type"] == "text":
                            full_response += event["content"]
                            text_placeholder.markdown(full_response + "▌")

            # Final static rendering after streaming completes
            text_placeholder.markdown(full_response)
            
            # Render sources expander below text if present
            if retrieved_sources:
                render_sources_expander(retrieved_sources)

            # Save full assistant response into session_state history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_sources
            })

        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
























# import requests

# url = "http://localhost:8000/query"

# payload = {
#     "query": "What is the IP address of Margaret Miller?"
# }

# with requests.post(url, json=payload, stream=True) as response:
#     response.raise_for_status()

#     for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
#         if chunk:
#             print(chunk, end="", flush=True)

# print()