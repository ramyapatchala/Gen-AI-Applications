import anthropic
import streamlit as st
import os
from dotenv import load_dotenv

question = "Why are LLMs danger to society?"

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def do_mistral(model):
  api_key = st.secrets['mistral_key']
  client = MistralClient(api_key = api_key)
  messages_to_LLM = [
    ChatMessage(role ='system', content = system_message),
    ChatMessage(role ='user', content = question)
  ]
  chat_response = client.chat(
    model=model,
    messages=messages_to_LLM,
  )
  return chat_response.choices[0].message.content

model='mistral_large_latest'
content = do_mistral(model)
st.write(content)
