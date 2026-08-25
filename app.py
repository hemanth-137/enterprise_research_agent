import streamlit as st
import requests

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")
st.title("RAG Assistant")
st.caption("A precise and strictly grounded QA assistant backed by Llama 3.1")

BACKEND_URL = "http://localhost:8000/query"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def stream_backend_response(user_query):
    try:
        payload = {"query": user_query}
        with requests.post(BACKEND_URL, json=payload, stream=True) as response:
            response.raise_for_status() 
            
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except requests.exceptions.ConnectionError:
        yield "⚠️ **Error:** Could not connect to the backend. Is FastAPI running?"
    except Exception as e:
        yield f"⚠️ **Error:** An unexpected error occurred: {str(e)}"

if prompt := st.chat_input("Ask a question..."):
    
    st.chat_message("user").markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_backend_response(prompt))
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})