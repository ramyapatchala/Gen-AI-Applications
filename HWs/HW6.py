import streamlit as st
import openai
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import json
import time

# Function to verify OpenAI API key
def verify_openai_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Vector DB functions
def add_to_collection(collection, text, filename):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response['data'][0]['embedding']
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding]
    )
    return collection

def setup_vectordb():
    db_path = "NewsBot_VectorDB"
    
    if not os.path.exists(db_path):
        st.info("Setting up vector DB for the first time...")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(
            name="NewsCollection",
            metadata={"hnsw:space": "cosine", "hnsw:M": 32}
        )
        
        # Load the CSV file containing news URLs
        news_df = pd.read_csv("Example_news_info_for_testing.csv")
        for index, row in news_df.iterrows():
            url = row['url']
            # Get content from the URL
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                collection = add_to_collection(collection, text, str(index))
            except Exception as e:
                st.warning(f"Could not fetch content from {url}: {e}")
        
        st.success(f"VectorDB setup complete with {len(news_df)} news articles!")
    else:
        st.info("VectorDB already exists. Loading from disk...")
        client = chromadb.PersistentClient(path=db_path)
        st.session_state.News_vectorDB = client.get_collection(name="NewsCollection")

def search_vectordb(query, k=3):
    if 'NewsBot_vectorDB' in st.session_state:
        collection = st.session_state.News_vectorDB
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response['data'][0]['embedding']
        
        with st.spinner('Retrieving information from the database...'):
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
st.title("News Reporting Bot")

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
        response = None
        
        if "interesting" in prompt.lower():
            results = search_vectordb("most interesting news")
            documents = results['documents']
            response_content = "Here are the most interesting news articles:\n" + "\n".join(documents)
        elif "find news about" in prompt.lower():
            topic = prompt.lower().split("find news about")[-1].strip()
            results = search_vectordb(topic)
            documents = results['documents']
            response_content = f"Here are news articles about '{topic}':\n" + "\n".join(documents)
        else:
            response_content = "I'm sorry, I can only help with finding interesting news or news about a specific topic."

        st.markdown(response_content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
