import cohere
import streamlit as st
cohere_key = st.secrets['cohere_key']

co = cohere.Client(
    api_key=cohere_key,
)

chat = co.chat(
    message="hello world!",
    model="command"
)

print(chat)
