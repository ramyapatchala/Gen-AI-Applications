import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import tiktoken

# Function definitions (read_webpage_from_url, calculate_tokens, truncate_messages_by_tokens) remain unchanged

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
        st.error(f"Error generating response: {str(e)}")
        return None    

st.title("My lab3 Question answering chatbot")

# Sidebar components remain unchanged

# Initialize the Gemini client
if 'client' not in st.session_state:
    api_key = st.secrets['gemini_key']
    genai.configure(api_key=api_key)
    st.session_state.client = genai.GenerativeModel('gemini-pro')

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Process URLs and combine documents (unchanged)

# Display chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message("assistant" if msg["role"] == "model" else "user")
    chat_msg.write(msg["parts"][0]["text"])

# Chat input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_message = {"role": "model", "parts": [{"text": f"Here are the documents to reference: {combined_document}"}]}
    messages_for_llm = [context_message] + st.session_state.messages

    # Apply conversation memory type
    if memory_type == "Buffer of 5 questions":
        messages_for_llm = messages_for_llm[-11:]  # System message + last 5 Q&A pairs
    elif memory_type == "Conversation summary":
        messages_for_llm = [context_message, messages_for_llm[-1]]
    else:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        messages_for_llm = truncate_messages_by_tokens(messages_for_llm, 5000, encoding)
    
    # Generate response using Gemini API
    client = st.session_state.client
    response = generate_gemini_response(client, messages_for_llm, prompt)

        
        # Add assistant response to chat history
    st.session_state.messages.append({"role": "model", "parts": [{"text": response}]})
