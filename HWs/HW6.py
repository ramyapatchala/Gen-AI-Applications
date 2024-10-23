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

# Function to calculate recency score
def calculate_recency_score(date_str):
    article_date = datetime.fromisoformat(date_str)  # Assuming date is in ISO format
    today = datetime.today()
    delta_days = (today - article_date).days
    return max(0, 1 / (delta_days + 1))  # The more recent, the higher the score

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
    keywords = ["legal", "lawsuit", "regulation", "merger", "acquisition", "court", "law", "contract"]

    # Generate embeddings for keywords
    openai_client = OpenAI(api_key=st.secrets['key1'])
    keyword_embeddings = []
    for keyword in keywords:
        response = openai_client.embeddings.create(
            input=keyword,
            model="text-embedding-3-small"
        )
        keyword_embeddings.append(response.data[0].embedding)

    # Combine keyword embeddings into a single embedding (mean of all keywords)
    combined_embedding = [sum(x) / len(x) for x in zip(*keyword_embeddings)]

    if 'News_Bot_VectorDB' in st.session_state:
        collection = st.session_state.News_Bot_VectorDB
        
        # Retrieve documents using combined keyword embeddings
        results = collection.query(
            query_embeddings=[combined_embedding],
            include=['documents', 'metadatas'],
            n_results=100  # Adjust this number as needed
        )
        
        interesting_articles = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            date_str = meta['date']
            keyword_frequency = calculate_keyword_frequency(doc, keywords)
            recency_score = calculate_recency_score(date_str)
            interesting_score = keyword_frequency * 0.7 + recency_score * 0.3  # Weighted score
            
            interesting_articles.append((date_str, doc, meta['id'], interesting_score))
        
        # Sort by the interesting score
        interesting_articles.sort(key=lambda x: x[3], reverse=True)
        
        return interesting_articles[:3]  # Return top 3 most interesting articles
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
            interesting_news = find_most_interesting_news()
            if interesting_news:
                formatted_results = [
                    f"{i + 1}. {doc} (Published on {date}) - [Link]({url})"
                    for i, (date, doc, url, _) in enumerate(interesting_news)
                ]
                response_content = "Here are the most interesting news articles:\n" + "\n".join(formatted_results)
            else:
                response_content = "No interesting news articles found."
        elif "find news about" in prompt.lower():
            topic = prompt.lower().split("find news about")[-1].strip()
            results = search_vectordb(topic)
            sorted_results = sort_results_by_date(results)

            formatted_results = [
                f"{i + 1}. {document} (Published on {date}) - [Link]({url})"
                for i, (date, document, url) in enumerate(sorted_results)
            ]
            response_content = f"Here are news articles about '{topic}':\n" + "\n".join(formatted_results)
        else:
            response_content = "I'm sorry, I can only help with finding interesting news or news about a specific topic."

        st.markdown(response_content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
