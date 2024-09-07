import anthropic
import streamlit as st
import os
from dotenv import load_dotenv

question = "Why are LLMs danger to society?"

def do_anthropic(model_to_use):
  api_key = st.secrets['claude_key']
  if model_to_use == 'sonnet':
    model= "claude-3-sonnet-20240229"
  elif model_to_use == 'haiku':
    model="claude-3-haiku-20230307"
  else:
    model = "claude-3-opus-20240229"
      
  client = anthropic.Anthropic(api_key = api_key)
  msg_to_llm=[
      {"role":"user", "content":[{"type":"text", "text":question}]}
    ]
  msg = client.messages.create(
      model = model,
      max_tokens = 1500,
      temperature = 0,
      messages = msg_to_llm
    )
  data = message.context[0].text
  return data

model = 'sonnet'
which_LLM = "Anthropic--"+str(model)
content = do_anthropic(model)
st.write('sonnet')
st.write(content)


model = 'haiku'
which_LLM = "Anthropic--"+str(model)
content = do_anthropic(model)
st.write('haiku')
st.write(content)


model = 'opus'
which_LLM = "Anthropic--"+str(model)
content = do_anthropic(model)
st.write('opus')
st.write(content)
