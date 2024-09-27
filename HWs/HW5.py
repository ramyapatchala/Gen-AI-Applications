import streamlit as st
import openai
import os
from PyPDF2 import PdfReader
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import tiktoken
from bs4 import BeautifulSoup
import requests

# Function to read webpage content from a URL
def read_webpage_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        document = " ".join([p.get_text() for p in soup.find_all("p")])
        return document
    except requests.RequestException as e:
        st.error(f"Error reading webpage from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing the webpage: {e}")
        return None

# Function to calculate tokens
def calculate_tokens(messages):
    """Calculate total tokens for a list of messages."""
    total_tokens = 0
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    for msg in messages:
        total_tokens += len(encoding.encode(msg['content']))
    return total_tokens

# Function to verify OpenAI API key
def verify_openai_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Function to generate summary using OpenAI
def generate_openai_response(client, messages, model):
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        return stream
    except Exception as e:
        st.error(f"Error generating response: {e}", icon="❌")
        return None

# Vector DB functions
def add_to_collection(collection, text, filename):
    openai_client = OpenAI(api_key = st.secrets['key1'])
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
    db_path = "HW4_VectorDB"
    
    # Check if vector DB exists on disk
    if not os.path.exists(db_path):
        st.info("Setting up vector DB for the first time...")
        
        # Initialize the ChromaDB client with persistence
        client = chromadb.PersistentClient(path=db_path)
        
        # Create or get the collection
        collection = client.get_or_create_collection(
            name="HW4Collection",
            metadata={"hnsw:space": "cosine", "hnsw:M": 32}
        )
        
        su_orgs_path = os.path.join(os.getcwd(), "HWs/su_orgs/")
        html_files = [f for f in os.listdir(su_orgs_path) if f.endswith('.html')]
        
        for html_file in html_files:
            file_path = os.path.join(su_orgs_path, html_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                collection = add_to_collection(collection, text, html_file)
        
        st.success(f"VectorDB setup complete with {len(html_files)} HTML files!")
    else:
        # If it already exists, just load it
        st.info("VectorDB already exists. Loading from disk...")
        client = chromadb.PersistentClient(path=db_path)
        st.session_state.HW4_vectorDB = client.get_collection(name="HW4Collection")


def query_vectordb(client, query, k=3):
    if 'HW4_vectorDB' in st.session_state:
        collection = st.session_state.HW4_vectorDB
        response = client.embeddings.create(
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

# Streamlit App
st.title("Interactive Data Search Chatbot")

# Sidebar: LLM provider selection
llm_provider = st.sidebar.selectbox("Choose LLM:", options=["OpenAI GPT-4O"])

# API key verification
openai_api_key = st.secrets["key1"]
client, is_valid, message = verify_openai_key(openai_api_key)

if is_valid:
    st.sidebar.success(f"OpenAI API key is valid!", icon="✅")
else:
    st.sidebar.error(f"Invalid OpenAI API key: {message}", icon="❌")
    st.stop()

# Set up VectorDB
setup_vectordb()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "system"
    with st.chat_message(role):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about iSchool student organizations?"):
    # Add user message to chat history
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Query VectorDB for relevant documents
    results = query_vectordb(client, prompt)
    if results:
        context = " ".join([doc for doc in results['documents'][0]])
        context_message = {"role": "system", "content": f"Relevant information: {context}"}
    else:
        context_message = {"role": "system", "content": "No specific context found."}

    messages_for_llm = [context_message] + st.session_state.messages

    # Generate response using OpenAI
    model = "gpt-4o-mini"
    full_response = ""
    message_placeholder = st.empty()
    stream = generate_openai_response(client, messages_for_llm, model)
    if stream:
      for chunk in stream:
        if chunk.choices[0].delta.content is not None:
          full_response += chunk.choices[0].delta.content
          message_placeholder.markdown(full_response + "▌")
      message_placeholder.markdown(full_response)


    st.session_state.messages.append({"role": "system", "content": full_response})
