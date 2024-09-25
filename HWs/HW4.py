import streamlit as st
from openai import OpenAI
import os
from bs4 import BeautifulSoup
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Initialize OpenAI client
if 'openai_client' not in st.session_state:
    api_key = st.secrets['key1']
    st.session_state.openai_client = OpenAI(api_key=api_key)

def add_to_collection(collection, text, filename):
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding]
    )
    return collection

def setup_vectordb():
    if 'HW4_vectorDB' not in st.session_state:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="HW4Collection",
            metadata={"hnsw:space": "cosine", "hnsw:M": 32}
        )
        
        su_orgs_path = os.path.join(os.getcwd(), "su_orgs")
        html_files = [f for f in os.listdir(su_orgs_path) if f.endswith('.html')]
        
        for html_file in html_files:
            file_path = os.path.join(su_orgs_path, html_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                collection = add_to_collection(collection, text, html_file)
        
        st.session_state.HW4_vectorDB = collection
        st.success(f"VectorDB setup complete with {len(html_files)} HTML files!")
    else:
        st.info("VectorDB already set up.")

def query_vectordb(query, k=3):
    if 'HW4_vectorDB' in st.session_state:
        collection = st.session_state.HW4_vectorDB
        openai_client = st.session_state.openai_client
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            include=['documents', 'distances', 'metadatas'],
            n_results=k
        )
        return results
    else:
        st.error("VectorDB not set up. Please set up the VectorDB first.")
        return None

def get_ai_response(query, context, model):
    openai_client = st.session_state.openai_client
    messages = [
        {"role": "system", "content": "You are a helpful assistant with knowledge about Syracuse University iSchool student organizations. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content

def main():
    st.title("iSchool Student Organizations Chatbot")

    setup_vectordb()

    # Sidebar for LLM selection
    llm_options = {
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
        "GPT-4": "gpt-4",
        "GPT-4 Turbo": "gpt-4-1106-preview"
    }
    selected_llm = st.sidebar.selectbox("Select LLM", list(llm_options.keys()))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about iSchool student organizations?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        results = query_vectordb(prompt)
        if results:
            context = " ".join([doc for doc in results['documents'][0]])
            response = get_ai_response(prompt, context, llm_options[selected_llm])
            
            final_response = f"(Using retrieved knowledge from documents)\n\n{response}"
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            with st.chat_message("assistant"):
                st.markdown(final_response)
                st.write("Related documents:")
                for i, doc_id in enumerate(results['ids'][0]):
                    st.write(f"{i+1}. {doc_id}")

        # Keep only the last 5 messages
        st.session_state.messages = st.session_state.messages[-10:]

main()
