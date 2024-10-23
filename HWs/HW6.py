import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to verify OpenAI API key
def verify_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Vector DB functions
def add_to_collection(collection, text, url):
    openai_client = OpenAI(api_key=st.secrets['key1'])
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[url],  # Store URL as the ID
        embeddings=[embedding],
        metadatas=[{"date": date}]
    )
    return collection

def setup_vectordb():
    db_path = "News_Bot_VectorDB"
    
    if not os.path.exists(db_path):
        st.info("Setting up vector DB for the first time...")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(
            name="NewsBotCollection",
            metadata={"hnsw:space": "cosine", "hnsw:M": 32}
        )
        
        # Load the CSV file containing news URLs
        news_df = pd.read_csv("HWs/Example_news_info_for_testing.csv")
        for _, row in news_df.iterrows():
            text = row['Document']
            url = row['URL']
            date = row['Date'] 
            add_to_collection(collection, text, url, date)  # Add document and URL to the collection
        
        st.success(f"VectorDB setup complete with {len(news_df)} news articles!")
    else:
        st.info("VectorDB already exists. Loading from disk...")
        client = chromadb.PersistentClient(path=db_path)
        st.session_state.News_Bot_VectorDB = client.get_collection(name="NewsBotCollection")

def search_vectordb(query, k=3):
    if 'News_Bot_VectorDB' in st.session_state:
        collection = st.session_state.News_Bot_VectorDB
        openai_client = OpenAI(api_key=st.secrets['key1'])
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            include=['documents', 'metadatas'],  # Exclude distances for simplicity
            n_results=k
        )
        st.write(results)
        sorted_results = sorted(
            zip(results['documents'][0], results['metadatas'], results['ids'][0]), 
            key=lambda x: (x[1]['date'], x[0]),  # Sort first by date, then by document relevance
            reverse=True  # Sort in descending order
        )
        return sorted_results
    else:
        st.error("VectorDB not set up. Please set up the VectorDB first.")
        return None

# Streamlit App
st.title("News Reporting Bot")

# API key verification
openai_api_key = st.secrets["key1"]
client, is_valid, message = verify_openai_key(openai_api_key)

if is_valid:
    st.sidebar.success("OpenAI API key is valid!", icon="✅")
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
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about the news?"):
    msg = {"role": "user", "content": prompt}
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(msg)
    
    # Generate response using OpenAI
    with st.chat_message("assistant"):
        response_content = None
        
        if "interesting" in prompt.lower():
            results = search_vectordb("most interesting news")
            urls = results['metadatas']  # Extract URLs from the results
            response_content = "Here are the most interesting news articles:\n" + "\n".join(urls)
        elif "find news about" in prompt.lower():
            topic = prompt.lower().split("find news about")[-1].strip()
            results = search_vectordb(topic)
            #st.write(results)
            #urls = results['metadatas']  # Extract URLs from the results
            formatted_results = []
            for i, document in enumerate(results['documents'][0]):
                url = results['ids'][0][i]
                formatted_results.append(f"{i + 1}. {document} To be continued ({url})")
            
            # Joining the formatted results into a single string
            #output = "\n".join(formatted_results)
            response_content = f"Here are news articles about '{topic}':\n" + "\n".join(formatted_results)
        else:
            response_content = "I'm sorry, I can only help with finding interesting news or news about a specific topic."

        st.markdown(response_content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
