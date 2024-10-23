import streamlit as st
from openai import OpenAI
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
        client = OpenAI(api_key=api_key)
        client.models.list()
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Vector DB functions
def add_to_collection(collection, text, filename):
    openai_client = OpenAI(api_key = st.secrets['key1'])
    response = openai_client.embeddings.create(
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
    db_path = "News_Bot_VectorDB"
    
    if os.path.exists(db_path):
        st.info("Setting up vector DB for the first time...")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(
            name="NewsBotCollection",
            metadata={"hnsw:space": "cosine", "hnsw:M": 32}
        )
        
        # Load the CSV file containing news URLs
        news_df = pd.read_csv("HWs/Example_news_info_for_testing.csv")
        for index, row in news_df.iterrows():
            url = row['URL']  # Ensure you use the correct column name
            # Get content from the URL
            try:
                st.write(url)
                response = requests.get(url)
                response.raise_for_status()  # Check if the request was successful
                soup = BeautifulSoup(response.content, 'html.parser')
                content = soup.find_all('p')  # Assuming the article is in <p> tags
                text = ""
                for paragraph in content:
                    text += paragraph.get_text()  # Concatenating the text
                collection = add_to_collection(collection, text, str(index))
            except requests.exceptions.RequestException as e:
                st.warning(f"Could not fetch content from {url}: {e}")

        
        st.success(f"VectorDB setup complete with {len(news_df)} news articles!")
    else:
        st.info("VectorDB already exists. Loading from disk...")
        client = chromadb.PersistentClient(path=db_path)
        st.session_state.News_Bot_vectorDB = client.get_collection(name="NewsBotCollection")

def search_vectordb(query, k=3):
    if 'NewsBot_vectorDB' in st.session_state:
        collection = st.session_state.News_Bot_vectorDB
        openai_client = OpenAI(api_key = st.secrets['key1'])
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
        response_content = None
        
        if "interesting" in prompt.lower():
            results = search_vectordb("most interesting news")
            documents = results['documents']
            response_content = "Here are the most interesting news articles:\n" + "\n".join(documents)
        elif "find news about" in prompt.lower():
            topic = prompt.lower().split("find news about")[-1].strip()
            results = search_vectordb(topic)
            context = " ".join([doc for doc in results['documents']])
            response_content = f"Here are news articles about '{topic}':\n" + "\n".join(context)
        else:
            response_content = "I'm sorry, I can only help with finding interesting news or news about a specific topic."

        st.markdown(response_content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
