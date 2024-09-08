import os
import streamlit as st
from mistralai import Mistral

api_key = st.secrets['mistral_key']
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)
st.write(chat_response.choices[0].message.content)
