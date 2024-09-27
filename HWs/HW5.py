import streamlit as st
import openai
import os
import chromadb
from PyPDF2 import PdfReader
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Verify OpenAI API key
def verify_openai_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Function to set up the VectorDB
def setup_vectordb():
    db_path = "HW4_VectorDB"
    
    if not os.path.exists(db_path):
        st.info("Setting up vector DB for the first time...")
        client = chromadb.PersistentClient(path=db_path)
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
                add_to_collection(collection, text, html_file)

        st.success(f"VectorDB setup complete with {len(html_files)} HTML files!")
    else:
        st.info("VectorDB already exists. Loading from disk...")
        client = chromadb.PersistentClient(path=db_path)
        st.session_state.HW4_vectorDB = client.get_collection(name="HW4Collection")

# Function to add documents to the collection
def add_to_collection(collection, text, filename):
    response = openai.Embedding.create(
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

# Function to perform vector search in ChromaDB
def search_vectordb(query):
    if 'HW4_vectorDB' in st.session_state:
        collection = st.session_state.HW4_vectorDB
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            include=['documents', 'distances', 'metadatas'],
            n_results=3
        )
        return results['documents'][0] if results['documents'] else "No relevant information found."
    else:
        return "VectorDB not set up."

# OpenAI function calling setup
def create_openai_functions():
    functions = [
        {
            "name": "search_vectordb",
            "description": "Search the vector database for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search the vector database."
                    }
                },
                "required": ["query"]
            }
        }
    ]
    return functions

# Function to call the OpenAI API with function calling
def generate_llm_response(client, query, context=None):
    messages = [{"role": "user", "content": query}]
    functions = create_openai_functions()

    # If we already have context (from vector search), pass it to LLM
    if context:
        messages.append({"role": "system", "content": context})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=functions,
        function_call="auto"  # Let the model decide when to call the function
    )
    
    return response

# Streamlit App
st.title("HW5 Interactive Course/Club Search Chatbot")

# Sidebar: API key verification
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

# Chat input
if prompt := st.chat_input("What would you like to know about iSchool student organizations or courses?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate LLM response with function calling
    response = generate_llm_response(client, prompt)
    st.write("Step 1")
    if response.choices[0].finish_reason == "function_call":
        # The LLM decided to call the `search_vectordb` function
        function_call = response.choices[0].message["function_call"]
        function_name = function_call["name"]
        function_args = function_call["arguments"]
        st.write("Step 2")
        # Execute the function call
        if function_name == "search_vectordb":
            query = function_args["query"]
            context = search_vectordb(query)
            st.write("Step 3")
            # Re-generate the LLM response with the new context
            response = generate_llm_response(client, prompt, context=context)
            st.write("Step 4")
    # Display the final response
    st.write("Step 5")
    st.write(response.choices[0].message.content)
