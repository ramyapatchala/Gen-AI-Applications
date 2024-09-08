import streamlit as st
import os
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Define the question and system message
question = "Why are LLMs danger to society?"
system_message = "You are a helpful assistant."

def do_mistral(model):
    api_key = st.secrets['mistral_key']
    client = MistralClient(api_key=api_key)
    
    # Prepare messages
    messages_to_LLM = [
        ChatMessage(role='system', content=system_message),
        ChatMessage(role='user', content=question)
    ]
    
    try:
        # Get the chat response
        chat_response = client.chat(
            model=model,
            messages=messages_to_LLM
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Error retrieving response."

model = 'mistral_large_latest'
content = do_mistral(model)
st.write(content)
