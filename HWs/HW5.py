import streamlit as st
import openai
import os
import chromadb
from PyPDF2 import PdfReader
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Function to perform vector search in ChromaDB
def search_vectordb(query):
    if 'HW4_vectorDB' in st.session_state:
        collection = st.session_state.HW4_vectorDB
        response = openai.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            include=['documents', 'distances', 'metadatas'],
            n_results=3
        )
        return results['documents'][0]
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
    
    response = openai.ChatCompletion.create(
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
    
    if response.choices[0].finish_reason == "function_call":
        # The LLM decided to call the `search_vectordb` function
        function_call = response.choices[0].message["function_call"]
        function_name = function_call["name"]
        function_args = function_call["arguments"]
        st.write("Step 1")
        # Execute the function call
        if function_name == "search_vectordb":
            query = function_args["query"]
            context = search_vectordb(query)
            st.write("Step 2")
            # Re-generate the LLM response with the new context
            response = generate_llm_response(client, prompt, context=context)
    st.write("Step 2")
    # Display the final response
    st.write(response.choices[0].message['content'])

