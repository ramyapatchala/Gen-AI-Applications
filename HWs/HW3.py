import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import tiktoken

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
def calculate_tokens(messages, encoding):
    total_tokens = 0
    for msg in messages:
        total_tokens += len(encoding.encode(msg['content']))
    return total_tokens

# Function to truncate messages by tokens
def truncate_messages_by_tokens(messages, max_tokens, encoding):
    total_tokens = calculate_tokens(messages, encoding)
    while total_tokens > max_tokens and len(messages) > 1:
        messages.pop(0)
        total_tokens = calculate_tokens(messages, encoding)
    return messages

# Function to generate response using Cohere
def generate_gemini_response(client, messages, prompt):
    try:
        response = client.generate_content(
            contents=[*messages, {"role": "user", "parts": prompt}],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=1500,
            ),
            stream=True
        )
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None    

st.title("My lab3 Question answering chatbot")
# Sidebar: URL inputs
st.sidebar.header("URL Inputs")
url1 = st.sidebar.text_input("Enter the first URL:")
url2 = st.sidebar.text_input("Enter the second URL (optional):")

# Sidebar: LLM provider selection
st.sidebar.header("LLM Provider")
llm_provider = st.sidebar.selectbox(
    "Choose your LLM provider:",
    options=["Gemini"]
)

# Sidebar: Conversation memory type
st.sidebar.header("Conversation Memory")
memory_type = st.sidebar.radio(
    "Choose conversation memory type:",
    options=["Buffer of 5 questions", "Conversation summary", "Buffer of 5,000 tokens"]
)

# Initialize the Gemini client
if 'client' not in st.session_state:
    api_key = st.secrets['gemini_key']
    genai.configure(api_key=api_key)
    st.session_state.client = genai.GenerativeModel('gemini-pro')

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

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

# Display chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message("system" if msg["role"] == "model" else "user")
    chat_msg.write(msg["parts"])

# Chat input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "parts": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    context_message = {"role": "system", "parts": f"Here are the documents to reference: {combined_document}"}
    messages_for_llm = [context_message] + st.session_state.messages
    # Apply conversation memory type
    if memory_type == "Buffer of 5 questions":
        messages_for_llm = messages_for_llm[-11:]  # System message + last 5 Q&A pairs
    elif memory_type == "Conversation summary":
        messages_for_llm = [context_message, messages_for_llm[-1]]
    else:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        messages_for_llm = truncate_messages_by_tokens(messages_for_llm, 5000, encoding)
    
    # Prepare chat history for Gemini API
    chat_history = st.session_state.messages[:-1]  # Exclude the last message
    
    # Generate response using Gemini API
    client = st.session_state.client
    response = generate_gemini_response(client, chat_history, prompt)
    # Display assistant response
    with st.chat_message("system"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
    st.session_state.messages.append({"role": "system", "parts":full_response})
        
