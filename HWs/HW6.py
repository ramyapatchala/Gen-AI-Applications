import streamlit as st
from openai import OpenAI
import os
import chromadb
import pandas as pd
from datetime import datetime

# Function to verify OpenAI API key
def verify_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Function to calculate keyword frequency
def calculate_keyword_frequency(document, keywords):
    return sum(document.lower().count(kw) for kw in keywords)

# Vector DB functions
def add_to_collection(collection, text, url, date):
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

def find_most_interesting_news():
    keywords = ["legal", "lawsuit", "regulation", "merger", "acquisition", "court", "law", "contract", "legal precedent", "jurisdiction", "statutory", "litigation",
    "regulatory compliance", "intellectual property", "antitrust",]

    # Generate embeddings for keywords
    openai_client = OpenAI(api_key=st.secrets['key1'])
    keyword_embeddings = []
    for keyword in keywords:
        response = openai_client.embeddings.create(
            input=keyword,
            model="text-embedding-3-small"
        )
        keyword_embeddings.append(response.data[0].embedding)
    combined_embedding = [sum(x) / len(x) for x in zip(*keyword_embeddings)]

    if 'News_Bot_VectorDB' in st.session_state:
        collection = st.session_state.News_Bot_VectorDB
        
        # Retrieve documents using combined keyword embeddings
        results = collection.query(
            query_embeddings=[combined_embedding],
            include=['documents', 'metadatas'],
            n_results=3  # Adjust this number as needed
        )
        return results
    else:
        st.error("VectorDB not set up. Please set up the VectorDB first.")
        return None

def search_vectordb(topic):
    # Search functionality using topic keywords
    openai_client = OpenAI(api_key=st.secrets['key1'])
    response = openai_client.embeddings.create(
        input=topic,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    if 'News_Bot_VectorDB' in st.session_state:
        collection = st.session_state.News_Bot_VectorDB
        results = collection.query(
            query_embeddings=[embedding],
            include=['documents', 'metadatas'],
            n_results=3  # Adjust this number as needed
        )
        return results
    else:
        st.error("VectorDB not set up. Please set up the VectorDB first.")
        return None

def sort_results_by_date(results):
    # Sorting function for search results by date
    data = []
    for doc, metadata, url in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
        date = metadata.get('date', 'Unknown Date')
        data.append((date, doc, url))
    return sorted(data, key=lambda x: x[0], reverse=True)

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

# Functionality buttons
option = st.selectbox("Choose an option", ["Select an option", "Interesting News", "Find News About a Topic"])

# Handle each option based on user selection
if option == "Interesting News":
    st.subheader("Fetching the most interesting news articles...")
    results = find_most_interesting_news()
    if results:
        sorted_results = sort_results_by_date(results)
        documents = sorted_results["documents"][0]
        metadatas = sorted_results["metadatas"][0]
        ids = sorted_results["ids"][0]
        formatted_results = []
        for i, (doc, metadata, url) in enumerate(zip(documents, metadatas, ids)):
            date = metadata.get('date', 'Unknown Date')
            formatted_results.append(f"{i + 1}. {doc[:200]}... (Published on {date}) - [Link]({url})")
        response_content = "Here are the most interesting news articles:\n" + "\n".join(formatted_results)
        st.markdown(response_content)
    else:
        st.error("No interesting news found.")

elif option == "Find News About a Topic":
    topic = st.text_input("Enter a topic to find news about:")
    if st.button("Search"):
        if topic:
            st.subheader(f"Searching for news articles about '{topic}'...")
            results = search_vectordb(topic)
            if results:
                sorted_results = sort_results_by_date(results)
                formatted_results = [
                    f"{i + 1}. {document[:200]}... (Published on {date}) - [Link]({url})"
                    for i, (date, document, url) in enumerate(sorted_results)
                ]
                response_content = "Here are the news articles:\n" + "\n".join(formatted_results)
                st.markdown(response_content)
            else:
                st.error(f"No news articles found for the topic: {topic}")
        else:
            st.error("Please enter a valid topic.")
