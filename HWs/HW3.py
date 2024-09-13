import streamlit as st
import cohere
import requests
from bs4 import BeautifulSoup

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

# Function to generate response using Cohere
def generate_cohere_response(client, prompt, chat_history):
    try:
        stream = client.chat_stream(
            model='command',
            message=prompt,
            chat_history=chat_history,
            temperature=0,       
            max_tokens=1500
        )
        return stream
    except Exception as e:
        st.error(f"Error generating response: {e}", icon="❌")
        return None

st.title("My lab3 Question answering chatbot")

# Sidebar: URL inputs
st.sidebar.header("URL Inputs")
url1 = st.sidebar.text_input("Enter the first URL:")
url2 = st.sidebar.text_input("Enter the second URL (optional):")

# Process URLs
documents = []
if url1:
    doc1 = read_webpage_from_url(url1)
    if doc1:
        documents.append(doc1)
if url2:
    doc2 = read_webpage_from_url(url2)
    if doc2:
        documents.append(doc2)

# Combine documents
combined_document = "\n\n".join(documents)

# Initialize the Cohere client
if 'client' not in st.session_state:
    api_key = st.secrets['cohere_key']
    st.session_state.client = cohere.Client(api_key)

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": f"Here are the documents to reference: {combined_document}\nHow can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message("assistant" if msg["role"] == "system" else "user")
    chat_msg.write(msg["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare chat history for Cohere API
    chat_history = [
        {"role": msg["role"], "message": msg["content"]}
        for msg in st.session_state.messages[:-1]  # Exclude the last message
    ]
    
    # Generate response using Cohere API
    client = st.session_state.client
    stream = generate_cohere_response(client, prompt, chat_history)
    
    if stream:
        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for event in stream:
                if event.event_type == "text-generation":
                    full_response += event.text
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "system", "content": full_response})
        
        # Limit chat history to last 5 messages
        st.session_state.messages = st.session_state.messages[-5:]
    else:
        st.error("Failed to generate a response. Please try again.")
