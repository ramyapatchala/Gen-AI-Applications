import streamlit as st
import cohere
import requests
from bs4 import BeautifulSoup
import tiktoken
from openai import OpenAI
import google.generativeai as genai

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

# Function to verify OpenAI API key
def verify_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
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

def verify_cohere_key(api_key):
    try:
        client = cohere.Client(api_key)
        client.generate(prompt="Hello", max_tokens=5)
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Function to generate response using Cohere
def generate_cohere_response(client, messages):
    try:
        stream = client.chat_stream(
            model='command-r',
            message=messages[-1]['content'],
            chat_history=[{"role": m['role'], "message": m['content']} for m in messages[:-1]],
            temperature=0,       
            max_tokens=1500
        )
        return stream
    except Exception as e:
        st.error(f"Error generating response: {e}", icon="❌")
        return None

# Function to verify Gemini API key
def verify_gemini_key(api_key):
    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-pro')
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

def generate_gemini_response(client, messages, prompt):
    try:
        response = client.generate_content(
            contents=[*messages, {"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=1500,
            ),
            stream=True
        )
        return response
    except Exception as e:
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
    options=["OpenAI GPT-4O-Mini", "OpenAI GPT-4O", "Cohere", "Gemini"]
)

# Sidebar: Conversation memory type
st.sidebar.header("Conversation Memory")
memory_type = st.sidebar.radio(
    "Choose conversation memory type:",
    options=["Buffer of 5 questions", "Conversation summary", "Buffer of 5,000 tokens"]
)
# API key verification
if "OpenAI" in llm_provider:
    openai_api_key = st.secrets['key1']
    client, is_valid, message = verify_openai_key(openai_api_key)
    model = "gpt-4o-mini" if llm_provider == "OpenAI GPT-4O-Mini" else "gpt-4o"
elif "Cohere" in llm_provider:
    cohere_api_key = st.secrets['cohere_key']
    client, is_valid, message = verify_cohere_key(cohere_api_key)
else:
    gemini_api_key = st.secrets['gemini_key']
    client, is_valid, message = verify_gemini_key(gemini_api_key)

if is_valid:
    st.sidebar.success(f"{llm_provider} API key is valid!", icon="✅")
else:
    st.sidebar.error(f"Invalid {llm_provider} API key: {message}", icon="❌")
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

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
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "system"
    with st.chat_message(role):
        content = message.get("content") or  message.get("parts", [{}])[0].get("text", "")
        st.markdown(content)

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if llm_provider == "Gemini":
        context_message = {"role": "model", "parts": [{"text": f"Here are the documents to reference: {combined_document}"}]}
    else:
        context_message = {"role": "system", "content": f"Here are the documents to reference: {combined_document}"}
    messages_for_llm = [context_message] + st.session_state.messages

    # Apply conversation memory type
    if memory_type == "Buffer of 5 questions":
        messages_for_llm = messages_for_llm[-5:]  # System message + last 5 Q&A pairs
    elif memory_type == "Conversation summary":
        messages_for_llm = [context_message, messages_for_llm[-1]]
    else:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        messages_for_llm = truncate_messages_by_tokens(messages_for_llm, 5000, encoding)

    # Prepare chat history for Cohere API
    chat_history = [
        {"role": msg["role"], "message": msg["content"]}
        for msg in st.session_state.messages[:-1]  # Exclude the last message
    ]

    with st.chat_message("system"):
        message_placeholder = st.empty()
        full_response = ""
        if "OpenAI" in llm_provider:
            stream = generate_openai_response(client, messages_for_llm, model)
            if stream:
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        elif "Cohere" in llm_provider:
            stream = generate_cohere_response(client, messages_for_llm)
            if stream:
                for event in stream:
                    if event.event_type == "text-generation":
                        full_response += event.text
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        else:
            response = generate_gemini_response(client, messages_for_llm, prompt)
            if response:
                with st.chat_message("system"):
                    for chunk in response:
                        if chunk.text:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "system", "content": full_response})
