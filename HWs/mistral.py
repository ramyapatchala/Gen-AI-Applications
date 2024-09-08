import streamlit as st
import os
from dotenv import load_dotenv
from mistralai.client import MistralClient

# Define the question and system message
question = "Why are LLMs danger to society?"
system_message = "You are a helpful assistant."

def do_mistral(model):
    # Retrieve the API key from Streamlit secrets
    api_key = st.secrets.get('mistral_key', None)
    if not api_key:
        st.error("Mistral API key not found in Streamlit secrets.")
        return "Error: API key not found."

    client = MistralClient(api_key=api_key)

    # Prepare messages as dictionaries
    messages_to_LLM = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
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

# Define the model to use
model = 'mistral_large_latest'

# Main app code
st.title("Streamlit App with MistralAI")
content = do_mistral(model)
st.write(content)

